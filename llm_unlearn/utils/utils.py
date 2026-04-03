from transformers import TrainerCallback
import wandb
import torch
import random
import evaluate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import glob
import json

import copy
from torch.nn import DataParallel

from .tokenizer_resize import smart_tokenizer_and_embedding_resize


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors like past_key_values, but logits always come first!
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated by preprocess_logits_for_metrics but we need to shift the labels!
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


class ModelParamsLoggingCallback(TrainerCallback):
    def __init__(self):
        # Placeholder for the names of the randomly selected parameters!
        self.selected_param_names = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # If not already selected, choose 3 random parameters!
        if self.selected_param_names is None:
            all_param_names = [name for name, _ in model.named_parameters()]
            self.selected_param_names = random.sample(all_param_names, 3)

    def on_log(self, args, state, control, model=None, **kwargs):
        # Log the L2 norm of the randomly selected parameters!
        for name, param in model.named_parameters():
            if name in self.selected_param_names:
                wandb.log({f"{name}_l2_norm": torch.norm(param).item()})
                # wandb.log({f"{name}_require_grad": param.requires_grad})


def load_model_and_tokenizer(model_path_or_name, auto_device=False):
    # Be conservative with kwargs here: Colab/Kaggle frequently run older GPUs
    # (e.g., T4) and/or slightly different Transformers versions.
    # - pre-Ampere GPUs generally don't support bf16 well -> use fp16
    # - `use_flash_attention_2` isn't supported in all Transformers versions/models
    torch_dtype = torch.float16
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            torch_dtype = torch.bfloat16

    params = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        #"use_flash_attention_2": True,
        # Prefer safetensors when available, but we may have to fall back if
        # the local files are incomplete/corrupted (common with missing git-lfs).
        "use_safetensors": True,
    }
    if auto_device:
        params["device_map"] = "auto"

    def _looks_like_git_lfs_pointer(file_path: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                head = f.read(200)
            return b"git-lfs.github.com/spec" in head
        except Exception:
            return False

    def _raise_incomplete_weights_error(original_error: Exception) -> None:
        msg = (
            "Model weights appear to be incomplete/corrupted (often caused by missing Git LFS downloads).\n"
            f"Tried loading from: {model_path_or_name!r}\n"
            f"Original error: {type(original_error).__name__}: {original_error}\n\n"
            "If you cloned this repo on Colab/Kaggle, install Git LFS and pull the large files:\n"
            "  apt-get update -y\n"
            "  apt-get install -y git-lfs\n"
            "  git lfs install\n"
            "  cd /content/unlearning-plm && git lfs pull\n\n"
            "Then re-run. Alternatively, re-download the model using Hugging Face `snapshot_download`."
        )
        raise RuntimeError(msg) from original_error

    def _get_index_expected_shards(model_dir: str, index_filename: str) -> Tuple[str, List[str], int]:
        index_path = os.path.join(model_dir, index_filename)
        if not os.path.isfile(index_path):
            return index_path, [], 0
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        expected_total = int(index.get("metadata", {}).get("total_size", 0) or 0)
        weight_map = index.get("weight_map", {}) or {}
        shard_names = sorted(set(weight_map.values()))
        return index_path, shard_names, expected_total

    def _choose_weight_format_or_raise(model_dir: str) -> Dict:
        """If `model_dir` is a local folder, ensure shards exist and pick a viable format.

        Returns a dict of overrides to apply to `params` (e.g., {"use_safetensors": False}).
        """
        st_index, st_shards, st_total = _get_index_expected_shards(model_dir, "model.safetensors.index.json")
        pt_index, pt_shards, pt_total = _get_index_expected_shards(model_dir, "pytorch_model.bin.index.json")

        st_present = [s for s in st_shards if os.path.isfile(os.path.join(model_dir, s))]
        pt_present = [s for s in pt_shards if os.path.isfile(os.path.join(model_dir, s))]

        # If safetensors shards exist but are truncated, fail fast with a clear message.
        if st_shards and len(st_present) == len(st_shards) and st_total:
            actual_total = sum(os.path.getsize(os.path.join(model_dir, s)) for s in st_present)
            if actual_total < int(0.98 * int(st_total)):
                _raise_incomplete_weights_error(
                    RuntimeError(
                        f"Incomplete safetensors shards: have {actual_total} bytes, expected ~{int(st_total)} bytes"
                    )
                )

        # Prefer safetensors if shards exist; otherwise fall back to PyTorch shards if present.
        if st_shards and len(st_present) == len(st_shards):
            return {"use_safetensors": True}
        if pt_shards and len(pt_present) == len(pt_shards):
            return {"use_safetensors": False}

        # If index exists but shards are missing, error out early with a clear message.
        missing_msgs = []
        if st_shards and len(st_present) != len(st_shards):
            missing = [s for s in st_shards if s not in st_present]
            missing_msgs.append(
                f"- Missing safetensors shards referenced by {os.path.basename(st_index)}: {', '.join(missing[:5])}"
                + (" ..." if len(missing) > 5 else "")
            )
        if pt_shards and len(pt_present) != len(pt_shards):
            missing = [s for s in pt_shards if s not in pt_present]
            missing_msgs.append(
                f"- Missing PyTorch shards referenced by {os.path.basename(pt_index)}: {', '.join(missing[:5])}"
                + (" ..." if len(missing) > 5 else "")
            )

        # If neither index exists, let transformers handle remote downloads.
        if not (os.path.isfile(st_index) or os.path.isfile(pt_index)):
            return {}

        # If at least one index exists but shards aren't here, it's almost certainly an incomplete checkout.
        total_hint = max(st_total, pt_total)
        hint = (
            "Local model directory has index file(s) but not the actual weight shards.\n"
            + (f"Expected total size: ~{total_hint/1e9:.2f} GB\n" if total_hint else "")
            + "\n".join(missing_msgs)
        )
        _raise_incomplete_weights_error(RuntimeError(hint))
        return {}

    # If loading from a local folder, pick a viable weight format early and
    # fail fast when shard files are missing (common when git-lfs wasn't used).
    if isinstance(model_path_or_name, str) and os.path.isdir(model_path_or_name):
        try:
            params.update(_choose_weight_format_or_raise(model_path_or_name))
        except RuntimeError:
            raise
        except Exception:
            # If the check fails unexpectedly, continue to normal loading and let it error.
            pass

    def _from_pretrained_with_retries() -> AutoModelForCausalLM:
        base = dict(params)

        # Try a small set of variants in a controlled way:
        # - flash-attn flag may not be supported by this Transformers version/model
        # - safetensors may be missing/corrupt; fall back to PyTorch shards
        variants = []
        variants.append(dict(base))
        if "use_flash_attention_2" in base:
            v = dict(base)
            v.pop("use_flash_attention_2", None)
            variants.append(v)
        if base.get("use_safetensors", None) is True:
            # Only fall back to PyTorch shards if they're actually present.
            can_fallback_to_pt = True
            if isinstance(model_path_or_name, str) and os.path.isdir(model_path_or_name):
                _pt_index, pt_shards, _pt_total = _get_index_expected_shards(
                    model_path_or_name, "pytorch_model.bin.index.json"
                )
                if pt_shards:
                    can_fallback_to_pt = all(
                        os.path.isfile(os.path.join(model_path_or_name, s)) for s in pt_shards
                    )
                else:
                    can_fallback_to_pt = False
            if can_fallback_to_pt:
                v = dict(base)
                v["use_safetensors"] = False
                variants.append(v)
                if "use_flash_attention_2" in base:
                    v2 = dict(v)
                    v2.pop("use_flash_attention_2", None)
                    variants.append(v2)

        last_error = None  # type: Optional[Exception]
        for local_params in variants:
            try:
                return AutoModelForCausalLM.from_pretrained(model_path_or_name, **local_params)
            except TypeError as e:
                last_error = e
                # Unsupported kwarg; continue to next variant.
                if "use_flash_attention_2" in str(e):
                    continue
                raise
            except Exception as e:
                last_error = e
                name = type(e).__name__
                text = str(e)
                if "safetensor" in text.lower() or name.lower() == "safetensorerror":
                    continue
                raise

        assert last_error is not None
        # If we exhausted variants, provide a better hint for common incomplete-weight cases.
        if isinstance(model_path_or_name, str) and os.path.isdir(model_path_or_name):
            for fp in glob.glob(os.path.join(model_path_or_name, "*.safetensors")):
                if _looks_like_git_lfs_pointer(fp):
                    _raise_incomplete_weights_error(last_error)
            for fp in glob.glob(os.path.join(model_path_or_name, "*.bin")):
                if _looks_like_git_lfs_pointer(fp):
                    _raise_incomplete_weights_error(last_error)

        # Re-raise with a clearer message if it looks like truncated weights.
        name = type(last_error).__name__
        text = str(last_error)
        if "safetensor" in text.lower() or name.lower() == "safetensorerror":
            _raise_incomplete_weights_error(last_error)
        raise last_error

    model = _from_pretrained_with_retries()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path_or_name,
        padding_side="right",
        trust_remote_code=True,
        model_max_length=4096,
    )
    smart_tokenizer_and_embedding_resize(tokenizer, model)
    return model, tokenizer