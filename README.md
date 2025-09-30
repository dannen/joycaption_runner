# JoyCaption: GPU-first batch captioner (installer + runner)

This repo ships a **single drop-in script** that (1) sets up a Python environment, (2) installs the JoyCaption model, and (3) captions **every image in a folder**. It defaults to **GPU + bfloat16** for speed/quality and is robust to tokenizer/template quirks in the upstream checkpoint.

- Model: `fancyfeast/llama-joycaption-beta-one-hf-llava`
- OS: Linux/macOS
- Python: 3.10+
- GPU: NVIDIA recommended (works on CPU too, slower)

---

## Contents

- `joycaption_install_and_run.sh` – one-shot installer + batch runner  
- `requirements.txt` – Python deps if you prefer manual venv setup

---

## Features

- **GPU-first defaults:** Installs CUDA PyTorch and runs in **bf16** if a GPU is available.
- **Correct image tokens & chat template:** Inserts the JoyCaption special image tokens and attaches a compatible chat template so generation works reliably.
- **Quantization options:** `bf16` (default), `8bit`, `4bit`, or `cpu`.
- **Fast downloads & stable memory:** Opt-in `hf_transfer` support and CUDA allocator tweaks baked in.
- **Batch-friendly:** Recursively scans a folder and writes sidecar `.txt` captions (mirrors subfolders).

---

## Quick start (recommended)

> Requires bash + Python 3.10+. If you’re in Docker, run the container with `--gpus all`.

```bash
# Make executable
chmod +x joycaption_install_and_run.sh

# Default run (GPU + bf16 + fast downloads + sensible CUDA settings)
./joycaption_install_and_run.sh --dir /path/to/images

This will create a local virtual environment at ./.joycaption_env, install everything, download the model the first time, and write captions to <images>/joycaptions.
Using a Python virtual environment (manual install)

If you’d rather manage the environment yourself:

# 1) Create & activate a venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install Python deps (torch installed separately, see below)
pip install -r requirements.txt

# 3) Install the correct PyTorch build
#   CUDA 12.4 wheels (works on modern NVIDIA drivers):
pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

#   CPU-only wheels (if you don’t have a GPU):
# pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

Now run the script as usual (it will detect the active venv):

./joycaption_install_and_run.sh --dir /path/to/images

Examples

1) Default (best balance on an NVIDIA GPU)

./joycaption_install_and_run.sh --dir ~/photos

2) Custom prompt + different output dir

./joycaption_install_and_run.sh \
  --dir ~/photos \
  --out ~/photos/captions_longform \
  --prompt "Write a straightforward, accurate caption in under 60 words."

3) Throughput mode (shorter generations)

./joycaption_install_and_run.sh --dir ~/photos --max-tokens 192

4) Lower VRAM (8-bit / 4-bit quantization)

./joycaption_install_and_run.sh --dir ~/photos --quant 8bit
./joycaption_install_and_run.sh --dir ~/photos --quant 4bit

5) Force CPU (slow, but universal)

./joycaption_install_and_run.sh --dir ~/photos --cpu

6) Limit to certain file types

./joycaption_install_and_run.sh --dir ~/photos --pattern "*.{jpg,jpeg,png}"

7) Choose a different CUDA wheel index

./joycaption_install_and_run.sh --dir ~/photos --cuda-url https://download.pytorch.org/whl/cu121

Script options (flags)

    --dir PATH (required): Folder containing images (recurses).

    --out PATH: Output folder for .txt captions (default: <dir>/joycaptions).

    --prompt TEXT: Captioning prompt (default: long, detailed description).

    --quant {bf16,8bit,4bit,cpu,auto}: Quantization (default: bf16 on GPU).

    --pattern GLOB: File glob (default: *.{jpg,jpeg,png,webp,bmp,tiff}).

    --max-tokens N: Generation length (default: 256).

    --clear-cache: Redownload model files (clears local HF cache for this model).

    --cpu: Force CPU install/run (overrides GPU defaults).

    --no-hf-transfer: Disable the fast download plugin if it causes issues.

    --hf-home PATH: Set/override the Hugging Face cache directory.

    --alloc-conf VALUE: Override PYTORCH_CUDA_ALLOC_CONF (default: expandable_segments:True).

    --cuda-connections N: Override CUDA_DEVICE_MAX_CONNECTIONS (default: 1).

    --cuda-url URL: Choose a specific PyTorch CUDA wheel index (default: cu124).

Recommended environment variables

These are already enabled by default in the script, but you can set them yourself too:

# Put the HF cache on a fast NVMe drive
export HF_HOME=/fast-nvme/hf-cache

# Faster model downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Smoother CUDA memory behavior
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

Output

For each image foo/bar.jpg, the script writes a caption to:

<out>/foo/bar.txt

Troubleshooting

    Script says Device: cpu | Quant: cpu but you have a GPU
    Ensure the venv has a CUDA PyTorch build and the process can see your GPU:

    source ./.joycaption_env/venv/bin/activate
    python - << 'PY'

import torch, sys
print("torch:", torch.version)
print("built_with_cuda:", torch.backends.cuda.is_built())
print("cuda_is_available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
PY

If `built_with_cuda=False` or `cuda_is_available=False`, reinstall torch with the right CUDA wheels
(`--extra-index-url https://download.pytorch.org/whl/cu124`), or run the container with `--gpus all`.

- **“Fast download using hf_transfer is enabled but package not available”**  
Either install it inside the venv (`pip install -U hf_transfer huggingface_hub`) or add `--no-hf-transfer`.

- **OOM (out of memory) on GPU**  
Try `--quant 8bit` or lower `--max-tokens` (e.g., `192`).

- **No images found**  
Double-check `--pattern` (brace globs are supported: `*.{jpg,png}`) and that `--dir` exists.

---

## Notes on licensing

- This repo’s script(s) are yours to license (MIT is common).  
- The JoyCaption model and any base components are subject to their **own licenses** on Hugging Face. Review and comply with those licenses before use.

---
