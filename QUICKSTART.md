# WikiForge-GPT Quick Start Guide 🚀

**Get up and running in 30 minutes**

---

## ✅ Pre-Flight Checklist

Before you begin, verify:

- [ ] Windows PC with NVIDIA GPU (RTX 4060 or similar)
- [ ] CUDA 12.6 installed (check with `nvidia-smi`)
- [ ] 23GB Wikipedia dump at `F:\wiki_raw\enwiki-latest-pages-articles.xml_2`
- [ ] At least 700GB free space on E: drive
- [ ] Anaconda or Miniconda installed

---

## 🎯 Step-by-Step Setup (30 minutes)

### Step 1: Create Environment (5 minutes)

Open Anaconda Prompt or terminal:

```bash
# Create conda environment
conda create -n wikiforge python=3.10 -y

# Activate environment
conda activate wikiforge
```

### Step 2: Install PyTorch with CUDA (10 minutes)

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Wait for installation to complete (~5-10 minutes)**

### Step 3: Clone/Setup Project (2 minutes)

```bash
# Navigate to project location
cd E:\WikiForge-GPT

# Copy the setup files you received:
# - settings.py
# - paths.py
# - requirements.txt
# - PROJECT_PLAN.md
# - README.md
```

### Step 4: Install Dependencies (10 minutes)

```bash
# Install all required packages
pip install -r requirements.txt
```

**Wait for installation to complete (~10 minutes)**

### Step 5: Initialize Project Structure (1 minute)

```bash
# Create all directories
python paths.py
```

**Expected Output**:
```
INITIALIZING WIKIFORGE-GPT PROJECT
Creating directory structure...
✓ Created: E:\WikiForge-GPT\data\processed
✓ Created: E:\WikiForge-GPT\data\tokenizer
[... more directories ...]
✓ PROJECT INITIALIZATION COMPLETE
✓ All critical paths are accessible!
```

### Step 6: Verify Hardware (2 minutes)

Create `scripts/verify_hardware.py`:

```python
import torch
import sys

print("="*80)
print("HARDWARE VERIFICATION")
print("="*80)

print(f"\nPython Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"✓ GPU computation test passed")
    print(f"✓ Setup complete - ready to start!")
else:
    print("✗ CUDA not available - check your PyTorch installation")

print("="*80)
```

Run it:
```bash
python scripts/verify_hardware.py
```

**Expected Output**:
```
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 4060 Laptop GPU
GPU Memory: 8.00 GB
✓ GPU computation test passed
✓ Setup complete - ready to start!
```

---

## 🎯 What's Next?

Now that your environment is set up, you have two paths:

### Option A: Follow the Full Project Plan (Recommended)

Read `PROJECT_PLAN.md` for detailed, step-by-step instructions.

**Timeline**: 6-8 weeks
**Coverage**: Complete end-to-end implementation
**Learning**: Deep understanding of every component

### Option B: Quick Prototype (Fast Path)

Skip to building a tiny model to validate the pipeline:

1. **Download Pre-processed Data** (if available)
2. **Use Pre-trained Tokenizer** (download or use HuggingFace)
3. **Train Stage 0** (12M parameter model)
4. **Validate Pipeline**

**Timeline**: 1 week
**Coverage**: Basic proof of concept
**Learning**: High-level understanding

---

## 📋 Development Checklist

Use this to track your progress:

### Phase 0: Setup ✅ (You are here!)
- [x] Environment created
- [x] PyTorch installed
- [x] Project structure initialized
- [x] Hardware verified

### Phase 1: Data Processing (Week 1)
- [ ] Implement Wikipedia XML parser
- [ ] Process 23GB dump
- [ ] Clean and filter articles
- [ ] Create train/val/test splits
- [ ] Generate data quality report

### Phase 2: Tokenizer (Week 2)
- [ ] Implement/use BPE tokenizer
- [ ] Train tokenizers (8K, 16K, 32K, 50K vocab)
- [ ] Evaluate tokenizer quality
- [ ] Prepare tokenized datasets

### Phase 3: Model Architecture (Week 2)
- [ ] Implement attention mechanism
- [ ] Implement transformer block
- [ ] Implement full GPT model
- [ ] Test forward/backward pass
- [ ] Verify parameter count

### Phase 4: Stage 0 Training (Week 3)
- [ ] Implement data loader
- [ ] Implement training loop
- [ ] Configure Stage 0 (12M params)
- [ ] Train for 48 hours
- [ ] Validate loss decrease
- [ ] Test text generation

### Phase 5-7: Scale Up (Weeks 4-8)
- [ ] Train Stage 1 (45M params, 96 hours)
- [ ] Train Stage 2 (125M params, 168 hours)
- [ ] Train Stage 3 (350M params, 336 hours)

### Phase 8: Deployment (Week 9)
- [ ] Comprehensive evaluation
- [ ] Build inference pipeline
- [ ] Create demo application
- [ ] Documentation complete

---

## 🚨 Common First-Day Issues

### Issue: "CUDA not available"

**Check**:
```bash
nvidia-smi  # Should show GPU
nvcc --version  # Should show CUDA 12.6
```

**Solution**:
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "Import errors"

**Solution**:
```bash
# Make sure you're in the right environment
conda activate wikiforge

# Reinstall problematic packages
pip install --upgrade [package-name]
```

### Issue: "Directories not created"

**Solution**:
```bash
# Run with verbose output
python -c "from paths import PATHS; PATHS.create_all()"

# Check if paths exist
python -c "from paths import validate_paths; validate_paths()"
```

---

## 💡 Pro Tips for Success

1. **Start Small**: Don't jump to Stage 3 immediately
2. **Use TensorBoard**: Monitor training in real-time
3. **Save Often**: Checkpoints are your safety net
4. **Read Logs**: They tell you what's happening
5. **Experiment**: Try different hyperparameters
6. **Document**: Keep notes on what works
7. **Be Patient**: Training takes time

---

## 📖 Essential Reading (First Week)

Before diving into code:

1. **Day 1-2**: Read "Attention Is All You Need" (Transformer paper)
2. **Day 3**: Watch Andrej Karpathy's "Let's build GPT" (2 hours)
3. **Day 4**: Read "The Illustrated Transformer" (Jay Alammar)
4. **Day 5**: Skim GPT-2 paper (focus on architecture)
5. **Day 6-7**: Review PyTorch tutorials (if needed)

---

## 🎯 First Week Goals

By the end of Week 1, you should have:

✅ **Technical**:
- Environment fully set up
- Wikipedia data processed
- Basic understanding of data pipeline

✅ **Knowledge**:
- Understand transformer architecture
- Know what tokenization is
- Familiar with PyTorch basics

✅ **Code**:
- XML parser implemented
- Data cleaning pipeline working
- Train/val/test splits created

---

## 📊 Quick Reference

### Key Files

| File | Purpose |
|------|---------|
| `settings.py` | Global configuration |
| `paths.py` | Path management (import this everywhere!) |
| `requirements.txt` | Dependencies |
| `PROJECT_PLAN.md` | Detailed implementation guide |
| `README.md` | Project overview |

### Key Directories

| Directory | Contains |
|-----------|----------|
| `data/` | All datasets |
| `src/` | Source code |
| `scripts/` | Executable scripts |
| `models/` | Model checkpoints |
| `logs/` | Training logs |

### Key Commands

```bash
# Activate environment
conda activate wikiforge

# Run training
python scripts/train_stage_0.py

# Monitor training
tensorboard --logdir logs/tensorboard

# Generate text
python scripts/generate_samples.py

# Check GPU usage
nvidia-smi
```

---

## 🎉 You're Ready!

You've completed the setup. Now:

1. **Read PROJECT_PLAN.md** for detailed next steps
2. **Start with Phase 1** (Data Processing)
3. **Join the journey** of building GPT from scratch!

---

## 💬 Quick Help

**Stuck?**
- Check `PROJECT_PLAN.md` for detailed instructions
- Review `README.md` for troubleshooting
- Look at `settings.py` for configuration options

**Have Questions?**
- Read the Transformer paper
- Watch Karpathy's tutorial
- Check PyTorch documentation

**Ready to Code?**
- Start with `src/data/wikipedia_parser.py`
- Follow the task breakdown in `PROJECT_PLAN.md`
- Test each component before moving forward

---

**Good luck! 🚀**

*Remember: Every expert was once a beginner. Take it one step at a time.*
