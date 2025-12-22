# Installation Guide for SG-CL Training

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **GPU**: Recommended (16GB+ VRAM) but not required
- **Storage**: ~10GB for model downloads

---

## 🚀 Quick Install

### Step 1: Install Core Dependencies

```bash
# Install PyTorch (GPU version - RECOMMENDED)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install HuggingFace libraries
pip install transformers peft accelerate

# Install utilities
pip install tqdm requests
```

**OR** install everything from requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 2: Install spaCy Model (for SID)

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

Expected output:
```
PyTorch: 2.x.x
Transformers: 4.x.x
PEFT: 0.x.x
```

---

## 🖥️ CPU-Only Installation

If you don't have a GPU:

```bash
# Install CPU version of PyTorch
pip install torch torchvision torchaudio

# Install other dependencies
pip install transformers peft accelerate tqdm requests spacy
python -m spacy download en_core_web_sm
```

⚠️ **Warning**: Training will be VERY slow on CPU (hours instead of minutes)

---

## 🧪 Test Your Installation

Run the minimal test:

```bash
python run_sgcl_experiment.py --dataset minimal --name install_test
```

This should:
- Download Phi-3 model (~7.5GB) on first run
- Run 2 tiny tasks
- Complete in ~5-10 minutes on GPU

---

## 📦 What Gets Installed

### Core Libraries (REQUIRED)

| Library | Size | Purpose |
|---------|------|---------|
| `torch` | ~2GB | PyTorch framework |
| `transformers` | ~500MB | HuggingFace models (Phi-3) |
| `peft` | ~50MB | LoRA fine-tuning |
| `accelerate` | ~100MB | Distributed training |

### Models (Downloaded on First Run)

| Model | Size | When Downloaded |
|-------|------|-----------------|
| Phi-3-mini-4k | ~7.5GB | First training run |
| en_core_web_sm | ~15MB | When spaCy used |

**Total**: ~10GB for everything

---

## 🔧 Troubleshooting

### Issue: "Import 'transformers' could not be resolved"

**Solution**:
```bash
pip install transformers
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Use smaller LoRA rank: `--lora-r 4`
2. Use CPU mode: Set `device="cpu"` in config
3. Use smaller model (future: add GPT-2 option)

### Issue: "Model download too slow"

**Solution**: Set HuggingFace cache:
```bash
export HF_HOME="/path/to/large/drive"  # Linux/Mac
set HF_HOME=D:\models  # Windows
```

### Issue: "pytest not found"

**Solution**:
```bash
pip install pytest pytest-cov
```

---

## ✅ Verify Complete Installation

Run all tests:

```bash
# Test SID
python -m pytest sid/test_sid.py -v

# Test Guardrails  
python -m pytest test_guardrail.py -v

# Test Training Integration (requires GPU or patience)
python run_sgcl_experiment.py --dataset minimal --name verify_install
```

All tests passing = Installation complete! ✓

---

## 📊 System Requirements

### Minimum (CPU mode)
- CPU: 4 cores
- RAM: 16GB
- Storage: 10GB
- Time: Hours per experiment

### Recommended (GPU mode)
- GPU: 16GB VRAM (RTX 4060 Ti, A4000, etc.)
- CPU: 8 cores
- RAM: 32GB
- Storage: 20GB
- Time: Minutes per experiment

### Optimal (Large experiments)
- GPU: 24GB+ VRAM (RTX 4090, A5000, etc.)
- CPU: 16 cores
- RAM: 64GB
- Storage: 50GB
- Time: Fast experimentation

---

## 🐳 Docker Alternative (Future)

For reproducible environment:

```bash
# TODO: Create Dockerfile
docker build -t sgcl .
docker run --gpus all -v $(pwd):/workspace sgcl python run_sgcl_experiment.py
```

---

## 📝 Quick Start After Installation

```bash
# 1. Quick test (5 minutes)
python run_sgcl_experiment.py --dataset minimal

# 2. Toy experiment (20 minutes)
python run_sgcl_experiment.py --dataset toy

# 3. Full experiment (hours)
python run_sgcl_experiment.py --dataset seca
```

---

## 🆘 Still Having Issues?

1. Check Python version: `python --version` (need 3.8+)
2. Update pip: `pip install --upgrade pip`
3. Try clean install: 
   ```bash
   pip uninstall torch transformers peft
   pip install -r requirements.txt
   ```
4. Check GPU: `nvidia-smi` (should show GPU)

---

**Installation Status**: ✅ Dependencies defined, ready to install
