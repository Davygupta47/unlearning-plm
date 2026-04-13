# =============================================================================
# LLM Unlearning on Kaggle Free Tier – Qwen2-1.5B + MovieLens Data
# =============================================================================
# Run this notebook cell-by-cell in Kaggle (T4 GPU, 58 GB RAM).
# No wandb key needed. No 8-GPU cluster needed.
# Steps:
#   1. Clone repo & install dependencies
#   2. Download Qwen2-1.5B model weights from HuggingFace
#   3. Prepare MovieLens tokenized datasets
#   4. Run unlearning  (gradient_ascent method)
#   5. Run evaluation  (perplexity on forget/general sets)
#   6. Run MIA         (Membership Inference Attack → AUC score)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — GPU Check & System Info
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, os, sys

def sh(cmd, **kw):
    """Run a shell command and print output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, **kw)
    return result.returncode

# Check available hardware
sh("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo 'No GPU detected'")
sh("free -h | grep Mem")
sh("df -h /kaggle/working | tail -1")

print(f"\nPython: {sys.version}")
print(f"CUDA available: ", end=""); sh("python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')\"")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Install Dependencies
# ─────────────────────────────────────────────────────────────────────────────
sh("""pip install -q \
    transformers==4.46.0 \
    datasets \
    evaluate \
    huggingface_hub \
    accelerate \
    scikit-learn \
    matplotlib \
    wandb \
    tqdm \
    requests \
    pandas \
    sentencepiece \
    protobuf
""")

print("✅ Dependencies installed.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Clone the Unlearning Repo
# ─────────────────────────────────────────────────────────────────────────────
WORKDIR = "/kaggle/working"
REPO_DIR = f"{WORKDIR}/unlearning-plm"

os.chdir(WORKDIR)

if not os.path.exists(REPO_DIR):
    sh("git clone https://github.com/Davygupta47/unlearning-plm.git")
else:
    print("Repo already cloned. Pulling latest changes...")
    sh("git pull", cwd=REPO_DIR)

os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# Install the llm_unlearn package in editable mode
sh("pip install -q -e .")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Download Qwen2-1.5B Model from HuggingFace Hub
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Qwen2-1.5B is ~3 GB. This takes ~5-10 minutes on Kaggle.
# We use snapshot_download which handles sharded safetensors correctly.

from huggingface_hub import snapshot_download

MODEL_ID   = "Qwen/Qwen2-1.5B"
MODEL_DIR  = f"{WORKDIR}/models/Qwen2-1.5B"

if not os.path.exists(os.path.join(MODEL_DIR, "tokenizer.json")):
    print(f"Downloading {MODEL_ID} to {MODEL_DIR} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.md", "*.svg", "*.gitattributes"],
    )
    print("✅ Model downloaded.")
else:
    print("✅ Model already present.")

sh(f"ls -lh {MODEL_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Prepare MovieLens Tokenized Datasets
# ─────────────────────────────────────────────────────────────────────────────
# This downloads MovieLens 1M (~24 MB) and converts ratings to text,
# then tokenizes into .pt files the pipeline expects.

os.chdir(REPO_DIR)

sh(f"""python llm_unlearn/utils/prepare_kaggle_dataset.py \
    --tokenizer_name_or_path {MODEL_DIR} \
    --domain movielens \
    --model_max_length 512
""")

sh("find ./tokenized_dataset -name '*.pt' | head -20")
print("✅ Datasets prepared.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Edit JSON Config Paths  (auto-patch for Kaggle)
# ─────────────────────────────────────────────────────────────────────────────
import json

def patch_json(path, updates):
    if not os.path.exists(path):
        print(f"  Config not found: {path}")
        return
    with open(path) as f:
        cfg = json.load(f)
    cfg.update(updates)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Patched: {path}")

UNLEARN_CFG = f"{REPO_DIR}/configs/unlearn_movielens.json"
EVAL_CFG    = f"{REPO_DIR}/configs/eval_movielens.json"
MIA_CFG     = f"{REPO_DIR}/configs/mia_movielens.json"

# Detect GPU capability for bf16 vs fp16
import torch
USE_BF16 = False
USE_FP16 = True
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)[0]
    if cap >= 8:          # Ampere+ (A100) supports bf16
        USE_BF16 = True
        USE_FP16 = False

# ── Unlearn config
patch_json(UNLEARN_CFG, {
    "target_model_name_or_path": MODEL_DIR,
    "output_dir": f"{WORKDIR}/output/movielens/Qwen2-1.5B/unlearn/gradient_ascent",
    "bf16": USE_BF16,
    "fp16": USE_FP16,
})

# ── Eval config (points to output of unlearning)
UNLEARNED_MODEL_DIR = f"{WORKDIR}/output/movielens/Qwen2-1.5B/unlearn/gradient_ascent"
patch_json(EVAL_CFG, {
    "model_name_or_path": UNLEARNED_MODEL_DIR,
    "output_dir": f"{WORKDIR}/output/movielens/Qwen2-1.5B-eval",
    "bf16": USE_BF16,
    "fp16": USE_FP16,
})

# ── MIA config
patch_json(MIA_CFG, {
    "model_name_or_path": UNLEARNED_MODEL_DIR,
    "output_dir": f"{WORKDIR}/output/movielens/Qwen2-1.5B-mia",
    "bf16": USE_BF16,
    "fp16": USE_FP16,
})

print(f"\nbf16={USE_BF16}, fp16={USE_FP16}")
print("✅ Configs patched for this Kaggle session.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Run Unlearning  (Gradient Ascent on the MovieLens forget set)
# ─────────────────────────────────────────────────────────────────────────────
# Expected time: ~20–40 minutes for 1 epoch on a T4 with bs=1, gas=16
# Memory: ~14–15 GB VRAM thanks to gradient_checkpointing + fp16

os.chdir(REPO_DIR)

# Disable wandb to avoid timeout issues
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

sh(f"python llm_unlearn/run_unlearn.py {UNLEARN_CFG}")

sh(f"ls -lh {WORKDIR}/output/movielens/Qwen2-1.5B/unlearn/gradient_ascent/")
print("✅ Unlearning complete.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Run Evaluation  (Perplexity on forget vs general sets)
# ─────────────────────────────────────────────────────────────────────────────
# Lower perplexity on forget set (after unlearning) = worse generation = more forgotten
# General set perplexity should stay roughly stable = model utility preserved

os.chdir(REPO_DIR)

sh(f"python llm_unlearn/run_eval.py {EVAL_CFG}")

print("\n📄 Eval results:")
sh(f"cat {WORKDIR}/output/movielens/Qwen2-1.5B-eval/forget_eval_results.json 2>/dev/null || echo ''")
sh(f"cat {WORKDIR}/output/movielens/Qwen2-1.5B-eval/general_eval_results.json 2>/dev/null || echo ''")
print("✅ Evaluation complete.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Run MIA  (Membership Inference Attack)
# ─────────────────────────────────────────────────────────────────────────────
# AUC close to 0.5 → attacker can't distinguish forget from non-member → good unlearning
# AUC close to 1.0 → model still memorizes forget data → unlearning failed

os.chdir(REPO_DIR)

sh(f"python llm_unlearn/run_mia.py {MIA_CFG}")

MIA_AUC_FILE = f"{WORKDIR}/output/movielens/Qwen2-1.5B-mia/auc.txt"
MIA_AUC_IMG  = f"{WORKDIR}/output/movielens/Qwen2-1.5B-mia/auc.png"

print("\n📊 MIA Results:")
sh(f"cat {MIA_AUC_FILE} 2>/dev/null || echo 'auc.txt not found'")

# Show ROC curve inline
try:
    from IPython.display import Image, display
    if os.path.exists(MIA_AUC_IMG):
        display(Image(MIA_AUC_IMG))
        print("✅ ROC curve displayed above.")
except Exception:
    print(f"ROC curve saved at: {MIA_AUC_IMG}")

print("✅ MIA complete. Full pipeline done!")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Summary
# ─────────────────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════╗
║         LLM Unlearning Pipeline Summary                  ║
╠══════════════════════════════════════════════════════════╣
║ Model    : Qwen2-1.5B (Qwen/Qwen2-1.5B)                  ║
║ Method   : Gradient Ascent                               ║
║ Domain   : MovieLens (user rating sentences)             ║
║ Platform : Kaggle Free Tier (T4 GPU, 58 GB RAM)          ║
╠══════════════════════════════════════════════════════════╣
║ Outputs                                                  ║
║   Unlearned model : output/movielens/Qwen2-1.5B/unlearn/...   ║
║   Eval results   : output/movielens/Qwen2-1.5B-eval/          ║
║   MIA results    : output/movielens/Qwen2-1.5B-mia/auc.txt    ║
║   ROC curve      : output/movielens/Qwen2-1.5B-mia/auc.png    ║
╠══════════════════════════════════════════════════════════╣
║ Interpretation                                           ║
║   MIA AUC ~ 0.5 → Unlearning SUCCESSFUL (no leak)       ║
║   MIA AUC ~ 1.0 → Unlearning FAILED (data still known)  ║
╚══════════════════════════════════════════════════════════╝
""")