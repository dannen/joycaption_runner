#!/usr/bin/env bash
# JoyCaption one-shot installer + batch captioner
# Defaults: GPU + bf16 + fast downloads + memory-friendly CUDA alloc
set -euo pipefail

# ---------------- Defaults you asked for ----------------
DEFAULT_QUANT="bf16"
DEFAULT_MAX_TOKENS=256
DEFAULT_ENABLE_HF_TRANSFER=1
DEFAULT_ALLOC_CONF="expandable_segments:True"
DEFAULT_CUDA_CONNECTIONS=1
DEFAULT_CUDA_INDEX="${TORCH_CUDA_URL:-https://download.pytorch.org/whl/cu124}"

# ---------------- Args ----------------
DIR=""; OUT=""; PROMPT="Write a long detailed description for this image."
QUANT="${DEFAULT_QUANT}"
PATTERN="*.{jpg,jpeg,png,webp,bmp,tiff}"
MAX_TOKENS="${DEFAULT_MAX_TOKENS}"
CLEAR_CACHE=0; FORCE_CPU=0
ENABLE_HF_TRANSFER="${DEFAULT_ENABLE_HF_TRANSFER}"
HF_HOME_OVERRIDE=""
ALLOC_CONF_OVERRIDE="${DEFAULT_ALLOC_CONF}"
CUDA_CONN_OVERRIDE="${DEFAULT_CUDA_CONNECTIONS}"
CUDA_INDEX_OVERRIDE="${DEFAULT_CUDA_INDEX}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DIR="${2:-}"; shift 2;;
    --out) OUT="${2:-}"; shift 2;;
    --prompt) PROMPT="${2:-}"; shift 2;;
    --quant) QUANT="${2:-}"; shift 2;;
    --pattern) PATTERN="${2:-}"; shift 2;;
    --max-tokens) MAX_TOKENS="${2:-}"; shift 2;;
    --clear-cache) CLEAR_CACHE=1; shift;;
    --cpu) FORCE_CPU=1; shift;;
    --no-hf-transfer) ENABLE_HF_TRANSFER=0; shift;;
    --hf-home) HF_HOME_OVERRIDE="${2:-}"; shift 2;;
    --alloc-conf) ALLOC_CONF_OVERRIDE="${2:-}"; shift 2;;
    --cuda-connections) CUDA_CONN_OVERRIDE="${2:-}"; shift 2;;
    --cuda-url) CUDA_INDEX_OVERRIDE="${2:-}"; shift 2;;
    -h|--help) sed -n '1,220p' "$0"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

[[ -z "${DIR}" ]] && { echo "ERROR: --dir is required"; exit 1; }
[[ ! -d "${DIR}" ]] && { echo "ERROR: --dir '${DIR}' does not exist"; exit 1; }
OUT="${OUT:-${DIR%/}/joycaptions}"

# ---------------- Fast defaults env ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${SCRIPT_DIR}/.joycaption_env"
VENV_DIR="${WORKDIR}/venv"
PY_SCRIPT="${WORKDIR}/joycaption_batch.py"
MODEL_ID="fancyfeast/llama-joycaption-beta-one-hf-llava"

# HF cache: use override, else keep existing, else local fast dir
if [[ -n "${HF_HOME_OVERRIDE}" ]]; then
  export HF_HOME="${HF_HOME_OVERRIDE}"
elif [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="${SCRIPT_DIR}/.hf-cache"
fi
mkdir -p "${HF_HOME}"

# Transfer accel default (can be disabled via --no-hf-transfer)
export HF_HUB_ENABLE_HF_TRANSFER="${ENABLE_HF_TRANSFER}"

# CUDA memory niceties
export PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF_OVERRIDE}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_CONN_OVERRIDE}"

mkdir -p "${WORKDIR}" "${OUT}"
if [[ ! -d "${VENV_DIR}" ]]; then python3 -m venv "${VENV_DIR}"; fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Auto-install hf_transfer if enabled
if [[ "${HF_HUB_ENABLE_HF_TRANSFER}" =~ ^(1|true|True)$ ]]; then
  python - <<'PY'
try:
    import hf_transfer  # noqa
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "hf_transfer", "huggingface_hub"])
PY
fi

# ---------------- PyTorch install (DEFAULT: GPU) ----------------
install_cpu_torch() {
  echo "[INFO] Installing CPU-only PyTorch wheels…"
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

install_cuda_torch() {
  local idx="${1}"
  echo "[INFO] Installing CUDA PyTorch wheels from: ${idx}"
  python -m pip install --extra-index-url "${idx}" torch torchvision torchaudio
}

if [[ "${FORCE_CPU}" -eq 0 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    set +e
    install_cuda_torch "${CUDA_INDEX_OVERRIDE}"
    CUDA_OK=$?
    set -e
    if [[ "${CUDA_OK}" -ne 0 ]]; then
      echo "[WARN] CUDA wheels failed; falling back to CPU wheels."
      install_cpu_torch
    fi
  else
    echo "[WARN] No NVIDIA GPU detected (nvidia-smi missing); installing CPU wheels."
    install_cpu_torch
  fi
else
  echo "[INFO] --cpu specified; installing CPU wheels."
  install_cpu_torch
fi

# ---------------- Core deps ----------------
python -m pip install \
  "transformers==4.46.3" \
  "tokenizers==0.20.3" \
  "accelerate==0.34.2" \
  "huggingface-hub>=0.24" \
  "safetensors" "sentencepiece" "pillow"
case "${QUANT}" in 8bit|4bit|auto) python -m pip install bitsandbytes || true;; esac

# Optional: nuke caches
if [[ "${CLEAR_CACHE}" -eq 1 ]]; then
  echo "[INFO] Clearing HF cache for ${MODEL_ID} …"
  rm -rf "${HF_HOME}/hub/models--fancyfeast--llama-joycaption-beta-one-hf-llava" || true
fi

# ---------------- Runner ----------------
cat > "${PY_SCRIPT}" << 'PYCODE'
import argparse, os, glob, sys, json
from pathlib import Path
from PIL import Image
import torch

# hf_transfer safeguard
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") in {"1","true","True"}:
    try:
        import hf_transfer  # noqa: F401
    except Exception as e:
        print(f"[WARN] HF_TRANSFER requested but not available ({e}); disabling.")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import hf_hub_download
from transformers.utils import logging as hf_logging
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, LlavaProcessor, LlavaForConditionalGeneration

MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
JC_IMAGE_TOKEN = "<|reserved_special_token_69|>"
JC_TOKEN_TRIPLE = "<|reserved_special_token_70|><|reserved_special_token_69|><|reserved_special_token_71|>"

def find_images(root:str, pattern:str):
    if "{" in pattern and "}" in pattern:
        brace = pattern[pattern.find("{")+1:pattern.find("}")]
        exts = [e.strip() for e in brace.split(",")]
        head = pattern[:pattern.find("{")]
        tail = pattern[pattern.find("}")+1:]
        files = []
        for e in exts:
            files += glob.glob(os.path.join(root, f"{head}{e}{tail}"), recursive=True)
        return sorted(files)
    return sorted(glob.glob(os.path.join(root, pattern), recursive=True))

def load_chat_template_text():
    try:
        raw = hf_hub_download(MODEL_ID, "chat_template.json")
        with open(raw, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("chat_template")
    except Exception:
        return None

def load_processor_robust():
    """Prefer AutoProcessor (brings tokenizer + proper special tokens + template)."""
    hf_logging.set_verbosity_error()
    last_err = None
    for kwargs in [dict(trust_remote_code=True), dict(trust_remote_code=True, use_fast=False)]:
        try:
            proc = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)
            if not getattr(proc, "chat_template", None):
                tmpl = load_chat_template_text()
                if tmpl:
                    proc.chat_template = tmpl
            return proc
        except Exception as e:
            last_err = e

    # Fallback: assemble manually
    tok = None
    for fast in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=fast)
            break
        except Exception as e:
            last_err = e
    if tok is None:
        raise RuntimeError(f"Failed to load processor/tokenizer: {last_err}")

    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    img_proc = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    proc = LlavaProcessor(image_processor=img_proc, tokenizer=tok)
    tmpl = load_chat_template_text()
    if tmpl:
        proc.chat_template = tmpl
    return proc

def ensure_image_tokens_present(convo_string):
    if JC_IMAGE_TOKEN not in convo_string:
        convo_string = convo_string.replace(
            "<|start_header_id|>user<|end_header_id|>\n\n",
            f"<|start_header_id|>user<|end_header_id|>\n\n{JC_TOKEN_TRIPLE}"
        )
    return convo_string

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--quant", choices=["bf16","8bit","4bit","cpu","auto"], default="bf16")
    ap.add_argument("--pattern", default="*.{jpg,jpeg,png,webp,bmp,tiff}")
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.quant != "cpu" else "cpu"
    load_kwargs = {}
    dtype = None
    if device == "cuda":
        if args.quant in ("auto","bf16"):
            try:
                dtype = torch.bfloat16
                load_kwargs = dict(torch_dtype=dtype, device_map=0, trust_remote_code=True)
                _ = torch.cuda.get_device_properties(0).total_memory
            except Exception:
                pass
        if (args.quant in ("auto","bf16")) and dtype is None:
            args.quant = "8bit"
        if args.quant == "8bit":
            load_kwargs = dict(load_in_8bit=True, device_map=0, trust_remote_code=True)
        elif args.quant == "4bit":
            load_kwargs = dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, device_map=0, trust_remote_code=True)
        elif args.quant == "bf16" and dtype is None:
            load_kwargs = dict(load_in_8bit=True, device_map=0, trust_remote_code=True)
            args.quant = "8bit"
    else:
        args.quant = "cpu"
        load_kwargs = dict(trust_remote_code=True)

    print(f"[INFO] Device: {device}  |  Quant: {args.quant}")
    print("[INFO] Loading processor/model…")
    processor = load_processor_robust()
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()

    # Generation IDs sanity
    if getattr(model, "generation_config", None):
        tok = processor.tokenizer
        if model.generation_config.pad_token_id is None and tok.pad_token_id is not None:
            model.generation_config.pad_token_id = tok.pad_token_id
        if model.generation_config.eos_token_id is None and tok.eos_token_id is not None:
            model.generation_config.eos_token_id = tok.eos_token_id

    files = find_images(args.dir, args.pattern)
    if not files:
        print(f"[WARN] No images found in {args.dir} matching {args.pattern}")
        return

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(files, 1):
        try:
            image = Image.open(fp).convert("RGB")
            convo = [
                {"role":"system","content":"You are a helpful image captioner."},
                {"role":"user","content": args.prompt},
            ]
            convo_str = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            convo_str = ensure_image_tokens_present(convo_str)

            inputs = processor(text=[convo_str], images=[image], return_tensors="pt")
            if device == "cuda":
                inputs = {k:v.to("cuda") for k,v in inputs.items()}
                if "pixel_values" in inputs and inputs["pixel_values"].dtype != torch.bfloat16:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    top_k=None,
                    use_cache=True,
                    suppress_tokens=None,
                )[0]
                gen_ids = gen_ids[inputs['input_ids'].shape[1]:]
                caption = processor.tokenizer.decode(
                    gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).strip()

            rel = os.path.relpath(fp, args.dir)
            out_path = out_dir / (Path(rel).with_suffix(".txt"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(caption, encoding="utf-8")
            print(f"[{i}/{len(files)}] {rel} -> {out_path}")
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user."); break
        except Exception as e:
            print(f"[ERROR] Failed on {fp}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
PYCODE

echo "[INFO] Starting captioning…"
python "${PY_SCRIPT}" \
  --dir "${DIR}" \
  --out "${OUT}" \
  --prompt "${PROMPT}" \
  --quant "${QUANT}" \
  --pattern "${PATTERN}" \
  --max_tokens "${MAX_TOKENS}"
echo "[OK] Captions written to: ${OUT}"
