from transformers import TrainerCallback
import wandb
import torch
import random
import evaluate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict
import os
import glob

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
        "use_flash_attention_2": True,
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

    def _from_pretrained_with_retries() -> AutoModelForCausalLM:
        local_params = dict(params)
        try:
            return AutoModelForCausalLM.from_pretrained(model_path_or_name, **local_params)
        except TypeError as e:
            if "use_flash_attention_2" in str(e):
                local_params.pop("use_flash_attention_2", None)
                return AutoModelForCausalLM.from_pretrained(model_path_or_name, **local_params)
            raise
        except Exception as e:
            # Typical when `.safetensors` files are LFS pointers or truncated.
            name = type(e).__name__
            text = str(e)
            if "safetensor" in text.lower() or name.lower() == "safetensorerror":
                # Fall back to PyTorch `.bin` shards when present.
                local_params["use_safetensors"] = False
                try:
                    return AutoModelForCausalLM.from_pretrained(model_path_or_name, **local_params)
                except Exception as e2:
                    # If this is a local folder, detect LFS pointer files to give a better hint.
                    if isinstance(model_path_or_name, str) and os.path.isdir(model_path_or_name):
                        for fp in glob.glob(os.path.join(model_path_or_name, "*.safetensors")):
                            if _looks_like_git_lfs_pointer(fp):
                                _raise_incomplete_weights_error(e)
                        for fp in glob.glob(os.path.join(model_path_or_name, "*.bin")):
                            if _looks_like_git_lfs_pointer(fp):
                                _raise_incomplete_weights_error(e2)
                    _raise_incomplete_weights_error(e2)
            raise

    model = _from_pretrained_with_retries()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path_or_name,
        padding_side="right",
        trust_remote_code=True,
        model_max_length=4096,
    )
    smart_tokenizer_and_embedding_resize(tokenizer, model)
    return model, tokenizer