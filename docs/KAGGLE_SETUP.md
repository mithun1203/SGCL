# Training SGCL on Kaggle GPU

This guide shows you how to run your SGCL experiments on Kaggle's free GPU (T4 or P100).

## Why Kaggle?

✅ **Free GPU access** - T4 or P100 GPUs (30 hours/week)  
✅ **Better than Colab** - More stable, longer sessions  
✅ **No authorization issues** - Direct file upload or datasets  
✅ **Easy downloads** - Export results directly  

## Quick Start (3 Methods)

### Method 1: Create a Kaggle Dataset (Recommended) ⭐

**Best for:** Multiple runs, sharing with team, persistence

1. **Create ZIP of your project:**
   ```powershell
   # In your SGCL folder
   Compress-Archive -Path * -DestinationPath sgcl_project.zip
   ```

2. **Upload to Kaggle Datasets:**
   - Go to https://kaggle.com/datasets
   - Click "New Dataset"
   - Upload `sgcl_project.zip`
   - Title: "SGCL Project"
   - Make it private
   - Click "Create"

3. **Create a new notebook:**
   - Go to https://kaggle.com/code
   - Click "New Notebook"
   - Upload `kaggle_training.ipynb` or create new
   - Settings → Accelerator → **GPU T4 x2**
   - Settings → Add Input → Search for "SGCL Project"
   - Click your dataset to add it

4. **Run the notebook:**
   - Execute cells in order
   - Your files will be at `/kaggle/input/sgcl-project/`
   - Results saved to `/kaggle/working/experiments/`

### Method 2: Upload Notebook + Manual Files

**Best for:** Quick testing, small projects

1. **Go to Kaggle:**
   - https://kaggle.com/code
   - Click "New Notebook"
   - Upload `kaggle_training.ipynb`

2. **Enable GPU:**
   - Settings → Accelerator → GPU T4 x2

3. **Upload files via UI:**
   - Click "Add Data" → "Upload"
   - Upload these files/folders:
     - `sgcl_training.py`
     - `sgcl_data_loader.py`
     - `run_mini_cpu_experiment.py`
     - `run_quick_test.py`
     - `knowledge_base.json`
     - `sid/` folder
     - `guardrail/` folder
     - `seca_v2.0/` folder

4. **Run cells** - Files will be in `/kaggle/working/`

### Method 3: Clone from GitHub

**Best for:** Version control, public projects

1. **Make your repo public** (or use Kaggle secrets for private)

2. **Create notebook** and add this cell at the top:
   ```python
   # Clone your SGCL repository
   !git clone https://github.com/mithun1203/SGCL.git
   %cd SGCL
   !ls -la
   ```

3. **Enable GPU** and run cells

## Training Speed Comparison

| Environment | Device | Mini Experiment (4 samples) | Full SeCA (320 samples) |
|------------|--------|---------------------------|------------------------|
| Your PC | CPU | ~30-60 minutes | ~10-20 hours |
| Kaggle | T4 GPU | **~2-3 minutes** | **~1-2 hours** |
| Kaggle | P100 GPU | **~1-2 minutes** | **~30-60 minutes** |

## Step-by-Step: First Run on Kaggle

1. **Compress your project:**
   ```powershell
   cd "C:\Users\naikm\OneDrive\Desktop\MITHUN NAIK\CAPSTONE\SGCL new"
   Compress-Archive -Path * -DestinationPath sgcl_project.zip
   ```

2. **Upload to Kaggle:**
   - Go to kaggle.com → Sign in
   - Datasets → New Dataset
   - Upload `sgcl_project.zip`
   - Name: "sgcl-project", Private

3. **Create notebook:**
   - Code → New Notebook
   - Upload `kaggle_training.ipynb`
   - Settings:
     - ✓ Accelerator: GPU T4 x2
     - ✓ Internet: ON (for pip installs)
     - ✓ Add Input: your "sgcl-project" dataset

4. **Run experiments:**
   - Run all cells sequentially
   - Check GPU is detected
   - Install dependencies
   - Verify files are present
   - Run mini experiment (~2-3 min)
   - View results

5. **Download results:**
   - Results are in `/kaggle/working/experiments/`
   - Use the download cell to create ZIP
   - Download from Output panel (right side)

## Kaggle Notebook Structure

```
kaggle_training.ipynb
├── Check GPU (verify T4/P100)
├── Install Dependencies (transformers, peft)
├── Load SGCL Files (from dataset or upload)
├── Verify Files (check all required files)
├── Run Quick Test (147K trainable params)
├── Run Mini Experiment (4 samples, SG-CL vs baseline)
├── View Results (comparison.json)
└── Download Results (ZIP for local analysis)
```

## Troubleshooting

### "No GPU detected"
- Settings → Accelerator → GPU T4 x2
- Click "Save" and restart kernel

### "Files not found"
- Check dataset is added: Settings → Input
- Or upload files manually: Add Data → Upload
- Verify path: `/kaggle/input/sgcl-project/` or `/kaggle/working/`

### "Out of memory"
- You're using GPT-2 (124M params) - should work fine
- If using Phi-3 (3.8B), reduce batch size or use smaller LoRA rank

### "Package not found"
- Make sure Internet is ON: Settings → Internet
- Run the dependency install cell again

## Files to Include

**Essential files:**
```
sgcl_training.py          # Core training logic
sgcl_data_loader.py       # Data loading
run_mini_cpu_experiment.py # Mini experiment runner
run_quick_test.py         # Environment verification
knowledge_base.json       # Knowledge base for guardrails
sid/                      # SID module
  ├── __init__.py
  ├── sid_module.py
  └── ...
guardrail/                # Guardrail system
  ├── __init__.py
  ├── guardrail_module.py
  └── ...
seca_v2.0/                # SeCA dataset
  └── tasks/
      ├── task1.json
      └── ...
```

## Next Steps

After your first successful run:

1. ✅ **Verify results** - Check SG-CL reduces conflicts vs baseline
2. ✅ **Scale up** - Run on full SeCA dataset (320 samples)
3. ✅ **Try different models** - Phi-3, Llama, etc.
4. ✅ **Tune hyperparameters** - LoRA rank, learning rate, etc.
5. ✅ **Collect metrics** - For your paper/report

## Useful Kaggle Tips

- **Save your work:** Notebooks auto-save, but commit versions regularly
- **Session limits:** 12 hours per session, 30 hours GPU/week (more than enough)
- **Multiple GPUs:** "GPU T4 x2" gives you 2 GPUs (data parallel training possible)
- **Persistent datasets:** Your uploaded datasets stay forever (notebooks timeout)
- **Sharing:** Make notebook public to share results with others

## Resources

- Kaggle Documentation: https://www.kaggle.com/docs
- Your notebook: Upload `kaggle_training.ipynb`
- GPU specs: T4 (16GB), P100 (16GB)
- Python 3.10, PyTorch pre-installed

---

**Ready to train?** 🚀

1. Compress project → ZIP
2. Upload to Kaggle Datasets
3. Create notebook with GPU
4. Add dataset as input
5. Run experiments
6. Download results

You'll have results in **under 10 minutes** from start to finish!
