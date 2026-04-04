# WikiForge-GPT 🚀

**Train GPT from scratch on Wikipedia data** - A complete implementation of the GPT architecture built from the ground up.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

WikiForge-GPT is a complete pipeline for training GPT-style language models from scratch using Wikipedia data. This project demonstrates:

- **Custom BPE Tokenizer** - Train your own vocabulary from scratch
- **Multi-Stage Training** - Progressive model scaling (Tiny → Small → Medium → Large)
- **Memory-Efficient Processing** - Handle massive datasets on consumer hardware
- **Production Training Loop** - Mixed precision, gradient accumulation, checkpointing

### 📊 What's Included

```
WikiForge-GPT/
├── scripts/              # Training pipeline scripts
│   ├── step1_train_tokenizer.py      # Custom BPE tokenizer
│   ├── step2_tokenize_dataset.py     # Dataset preprocessing
│   ├── step3_gpt_architecture.py     # GPT model from scratch
│   └── step4_train_model.py          # Full training loop
├── paths.py              # Centralized path management
├── settings.py           # Project configuration
└── README.md             # This file
```

## 🏗️ Architecture

### Stage 0: Tiny Model (Proof of Concept)
- **Parameters**: 5.3M
- **Vocabulary**: 8,000 tokens
- **Layers**: 4 transformer blocks
- **Context**: 256 tokens
- **Training Time**: ~4 hours on RTX 4060

### Future Stages
- **Stage 1 (Small)**: 45M params, 16K vocab
- **Stage 2 (Medium)**: 125M params, 32K vocab
- **Stage 3 (Large)**: 350M params, 50K vocab

## 💻 Hardware Requirements

### Minimum (Stage 0 - Tiny)
- **GPU**: RTX 3060 (8GB VRAM) or better
- **RAM**: 16GB
- **Storage**: 50GB free space
- **CPU**: 4+ cores

### Recommended (Stage 3 - Large)
- **GPU**: RTX 4090 (24GB VRAM)
- **RAM**: 32GB+
- **Storage**: 200GB+ SSD

## 📥 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/WikiForge-GPT.git
cd WikiForge-GPT
```

### 2. Create Virtual Environment
```bash
python -m venv .wikienv
# Windows
.wikienv\Scripts\activate
# Linux/Mac
source .wikienv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Wikipedia Data
Download the latest Wikipedia dump:
```bash
# English Wikipedia (latest)
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Extract (this will take time!)
bunzip2 enwiki-latest-pages-articles.xml.bz2
```

## 🚀 Quick Start

### Step 1: Train Custom Tokenizer
```bash
python scripts/step1_train_tokenizer.py
```
**Output**: Custom 8K vocabulary BPE tokenizer  
**Time**: ~15 minutes

### Step 2: Tokenize Dataset
```bash
python scripts/step2_tokenize_dataset.py
```
**Output**: 19.6M tokenized sequences (18.7 GB)  
**Time**: ~45 minutes

### Step 3: Test Architecture
```bash
python scripts/step3_gpt_architecture.py
```
**Output**: Verified 5.3M parameter GPT model  
**Time**: ~5 seconds

### Step 4: Train Model
```bash
python scripts/step4_train_model.py
```
**Output**: Trained GPT model with checkpoints  
**Time**: ~4 hours (100K steps)

## 📊 Training Results

### Stage 0 (Tiny Model)
```
Dataset: 2.88M Wikipedia articles
Training Tokens: 5 billion
Training Steps: 100,000
Final Loss: ~4.5
Validation Loss: ~4.2
```

### Sample Output
```
Step  50000/100000 ( 50.0%) | Loss: 5.2341 | LR: 2.12e-04 | 
156.3ms/step | Elapsed: 2h05m43s | ETA: 2h05m43s | VRAM: 3.2/8.0GB
```

## 📁 Project Structure

```
WikiForge-GPT/
├── data/
│   ├── tokenizer/           # Trained tokenizers
│   ├── training/            # Tokenized data (git-ignored)
│   └── processed/           # Cleaned Wikipedia (git-ignored)
├── models/
│   └── checkpoints/         # Model checkpoints (git-ignored)
├── scripts/
│   ├── step1_train_tokenizer.py
│   ├── step2_tokenize_dataset.py
│   ├── step3_gpt_architecture.py
│   └── step4_train_model.py
├── logs/                    # Training logs (git-ignored)
├── paths.py                 # Path configuration
├── settings.py              # Model & training config
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

Edit `settings.py` to customize:

```python
# Model Architecture (Stage 0)
vocab_size = 8000
n_layers = 4
n_heads = 4
d_model = 256
max_seq_length = 256

# Training
batch_size = 4
gradient_accumulation_steps = 8
learning_rate = 3e-4
max_steps = 100000
```

## 🎓 Key Features

### Custom Tokenizer
- **BPE Algorithm** - Industry-standard byte-pair encoding
- **Vocabulary Optimization** - Trained on your specific dataset
- **Special Tokens** - `<|endoftext|>`, `<|pad|>`, `<|unk|>`

### Memory-Efficient Training
- **Chunk Processing** - 10K articles at a time
- **Memory-Mapped Data** - Efficient loading of large datasets
- **Mixed Precision** - FP16 for 2x speed & memory savings

### Production Features
- **Gradient Accumulation** - Simulate larger batch sizes
- **Cosine LR Schedule** - Warmup + decay
- **Auto Checkpointing** - Save every 5K steps
- **Validation Tracking** - Best model selection
- **Detailed Logging** - ETA, VRAM, timing metrics

## 📈 Performance Metrics

### Training Speed (RTX 4060)
- **Steps/sec**: ~6.5
- **Tokens/sec**: ~13,000
- **Time to 100K steps**: ~4.3 hours

### Memory Usage
- **Model**: ~400 MB
- **Optimizer**: ~800 MB
- **Activations**: ~1.2 GB
- **Peak VRAM**: ~3.2 GB / 8 GB

## 🔧 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size in settings.py
batch_size = 2  # Instead of 4
gradient_accumulation_steps = 16  # Instead of 8
```

### Slow Training
```python
# Use mixed precision (enabled by default)
use_amp = True

# Reduce validation frequency
eval_interval = 2000  # Instead of 1000
```

## 📚 Learning Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Distributed training support
- [ ] More tokenizer options (WordPiece, Unigram)
- [ ] Inference optimization
- [ ] Model quantization
- [ ] Web interface for text generation

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Andrej Karpathy** - For educational resources on training GPT
- **OpenAI** - For GPT architecture inspiration
- **Hugging Face** - For tokenizers library
- **PyTorch** - For the deep learning framework

## 📞 Contact

- **Author**: Anupam
- **Hardware**: RTX 4060 Laptop, i5-12450H, 16GB RAM
- **Project**: WikiForge-GPT

---

⭐ **Star this repo** if you found it helpful!

Built with ❤️ for the open-source ML community
