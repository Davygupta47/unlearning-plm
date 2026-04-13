"""
prepare_kaggle_dataset.py
=========================
One-shot script to build and tokenize unlearning datasets for Kaggle.

Supports domains:
  - movielens  (downloaded from grouplens.org or synthetic fallback)
  - arxiv / github  (from HuggingFace hub: llmunlearn/unlearn_dataset)

Usage:
    python llm_unlearn/utils/prepare_kaggle_dataset.py \
        --tokenizer_name_or_path ./models/Yi-6B \
        --domain movielens

Output:
    ./tokenized_dataset/movielens/movielens_forget_500/normal/tokenized_dataset.pt
    ./tokenized_dataset/movielens/movielens_approximate_500/normal/tokenized_dataset.pt
    ./tokenized_dataset/movielens/movielens_retain_1k/normal/tokenized_dataset.pt
    ./tokenized_dataset/general/general_1k/normal/tokenized_dataset.pt
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from datasets import Dataset
from transformers import set_seed, AutoTokenizer

# Allow running from any directory inside the repo
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_unlearn.utils.chunk_tokenizer import tokenize


MODEL_MAX_LENGTH = 512   # Use 512 on Kaggle to save memory (original is 4096)
OUTPUT_DIR = "./tokenized_dataset"


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _rows_to_hf_dataset(rows):
    """Convert list of {'text': ...} dicts to a HuggingFace Dataset."""
    return Dataset.from_dict({"text": [r["text"] for r in rows]})


def _save(dataset_hf, tokenizer, save_path, model_max_length):
    print(f"  Tokenizing {len(dataset_hf)} samples -> {save_path}")
    tokenized = tokenize(dataset_hf, tokenizer, model_max_length)
    os.makedirs(save_path, exist_ok=True)
    pt_path = os.path.join(save_path, "tokenized_dataset.pt")
    torch.save(tokenized, pt_path)
    print(f"  Saved: {pt_path}  ({len(tokenized)} chunks)")
    return pt_path


def prepare_movielens(tokenizer, model_max_length=MODEL_MAX_LENGTH):
    from llm_unlearn.utils.movielens_data import build_movielens_splits

    print("\n=== Building MovieLens splits ===")
    forget_rows, approx_rows, retain_rows = build_movielens_splits(
        cache_dir="./ml_cache",
        n_forget=500,
        n_approx=500,
        n_retain=1000,
    )

    splits = {
        "movielens/movielens_forget_500/normal": forget_rows,
        "movielens/movielens_approximate_500/normal": approx_rows,
        "movielens/movielens_retain_1k/normal": retain_rows,
        # "general" slot reuses the retain set (used by run_eval.py)
        "general/general_1k/normal": retain_rows,
    }

    for rel_path, rows in splits.items():
        ds = _rows_to_hf_dataset(rows)
        save_path = os.path.join(OUTPUT_DIR, rel_path)
        _save(ds, tokenizer, save_path, model_max_length)


def prepare_arxiv_github(tokenizer, domain, model_max_length=MODEL_MAX_LENGTH):
    """Prepare arxiv or github splits from the official HF dataset."""
    from datasets import load_dataset

    if domain == "arxiv":
        splits_map = {
            "arxiv/arxiv_forget_500/normal":   ("arxiv", "forget"),
            "arxiv/arxiv_approximate_6k/normal": ("arxiv", "approximate"),
            "general/general_1k/normal":         ("general", "evaluation"),
        }
    else:  # github
        splits_map = {
            "github/github_forget_2k/normal":    ("github", "forget"),
            "github/github_approximate/normal":  ("github", "approximate"),
            "general/general_1k/normal":         ("general", "evaluation"),
        }

    hf_ds_name = "llmunlearn/unlearn_dataset"
    for rel_path, (name, split) in splits_map.items():
        print(f"\n=== Loading {hf_ds_name} [{name}/{split}] ===")
        raw = load_dataset(hf_ds_name, name=name, split=split)
        # Ensure only 'text' column
        if "text" not in raw.column_names and "content" in raw.column_names:
            raw = raw.rename_column("content", "text")
        keep = [c for c in raw.column_names if c != "text"]
        raw = raw.remove_columns(keep)
        save_path = os.path.join(OUTPUT_DIR, rel_path)
        _save(raw, tokenizer, save_path, model_max_length)


def main():
    parser = argparse.ArgumentParser(description="Prepare tokenized datasets for Kaggle.")
    parser.add_argument(
        "--tokenizer_name_or_path", "-t", required=True,
        help="Path to the Yi-6B (or other) tokenizer directory."
    )
    parser.add_argument(
        "--domain", default="movielens",
        choices=["movielens", "arxiv", "github"],
        help="Which domain to prepare. Default: movielens"
    )
    parser.add_argument(
        "--model_max_length", type=int, default=MODEL_MAX_LENGTH,
        help=f"Max token length per chunk. Default: {MODEL_MAX_LENGTH}"
    )
    args = parser.parse_args()

    set_seed(42)

    print(f"\n[prepare_kaggle_dataset] Loading tokenizer from: {args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        padding_side="right",
        trust_remote_code=True,
        model_max_length=args.model_max_length,
    )
    _ensure_pad_token(tokenizer)

    if args.domain == "movielens":
        prepare_movielens(tokenizer, args.model_max_length)
    else:
        prepare_arxiv_github(tokenizer, args.domain, args.model_max_length)

    print("\n[prepare_kaggle_dataset] Done! All datasets saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
