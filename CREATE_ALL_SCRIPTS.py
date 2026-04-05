"""
═══════════════════════════════════════════════════════════════════════════════
IMPROVED AUTOMATIC SCRIPT GENERATOR - WITH TIME TRACKING & PERPLEXITY
═══════════════════════════════════════════════════════════════════════════════

Enhanced features:
✅ Time elapsed tracking
✅ Time remaining (ETA) calculation
✅ Perplexity calculation at every 1000 steps
✅ Detailed JSONL logging with all metrics
✅ Progress bars and status updates

Usage:
    cd E:/WikiForge-GPT
    python CREATE_ALL_SCRIPTS_IMPROVED.py

Creates all 12 files with enhanced logging!
═══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path

# Base directory
BASE_DIR = Path("E:/WikiForge-GPT/scripts")

# =============================================================================
# STAGE CONFIGURATIONS
# =============================================================================

CONFIGS = {
    'stage_1': {
        'vocab_size': 16000,
        'n_layer': 8,
        'n_head': 8,
        'd_model': 512,
        'd_ff': 2048,
        'batch_size': 2,
        'grad_accum': 16,
        'params': '45M',
        'time_train': '14 hours',
        'loss': '2.5',
    },
    'stage_2': {
        'vocab_size': 32000,
        'n_layer': 12,
        'n_head': 12,
        'd_model': 768,
        'd_ff': 3072,
        'batch_size': 2,
        'grad_accum': 16,
        'params': '125M',
        'time_train': '48 hours',
        'loss': '2.0',
    },
    'stage_3': {
        'vocab_size': 50000,
        'n_layer': 16,
        'n_head': 16,
        'd_model': 1024,
        'd_ff': 4096,
        'batch_size': 1,
        'grad_accum': 32,
        'params': '350M',
        'time_train': '240 hours',
        'loss': '1.7',
        'use_grad_checkpoint': True,
    },
}

# =============================================================================
# SCRIPT TEMPLATES WITH ENHANCED LOGGING
# =============================================================================

def get_train_script_improved(stage, config):
    return f'''"""
{stage.upper()} - Step 3: Train {config['params']} Model
===============================================================================
Architecture: {config['n_layer']} layers, {config['n_head']} heads, {config['d_model']} d_model
Expected time: {config['time_train']}
Expected loss: ~{config['loss']}

Enhanced Features:
✅ Time elapsed tracking
✅ Time remaining (ETA) calculation
✅ Perplexity calculation every 1000 steps
✅ Detailed JSONL logging
✅ Progress tracking
"""

import sys
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime, timedelta

PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))
from paths import PATHS
from scripts.step3_gpt_architecture import GPTModel, GPTConfig


def format_time(seconds):
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{{seconds:.0f}}s"
    elif seconds < 3600:
        return f"{{seconds/60:.0f}}m {{seconds%60:.0f}}s"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{{hours}}h {{mins}}m"
    else:
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{{days}}d {{hours}}h"


def train_model():
    print("\\n" + "="*80)
    print("{stage.upper()} - TRAIN {config['params']} MODEL")
    print("="*80 + "\\n")
    
    # Configuration
    config_obj = GPTConfig(
        vocab_size={config['vocab_size']},
        n_layers={config['n_layer']},
        n_heads={config['n_head']},
        d_model={config['d_model']},
        d_ff={config['d_ff']},
        max_seq_length=256,
        dropout=0.1,
        pad_token_id=1,
        eos_token_id=0,
    )
    
    batch_size = {config['batch_size']}
    gradient_accumulation_steps = {config['grad_accum']}
    learning_rate = 3e-4
    max_steps = 100000
    warmup_steps = 2000
    min_lr = 3e-5
    
    log_interval = 100
    eval_interval = 1000
    save_interval = 5000
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📋 Configuration:")
    print(f"  Model: {config['params']} parameters")
    print(f"  Architecture: {config['n_layer']} layers, {config['n_head']} heads, {config['d_model']} d_model")
    print(f"  Vocab size: {config['vocab_size']:,}")
    print(f"  Batch size: {{batch_size}} × {{gradient_accumulation_steps}} = {{batch_size*gradient_accumulation_steps}}")
    print(f"  Learning rate: {{learning_rate}} → {{min_lr}}")
    print(f"  Max steps: {{max_steps:,}}")
    print(f"  Device: {{device}}")
    print()
    
    # Load data
    data_dir = PATHS.TRAINING_DATA / "{stage}_vocab{config['vocab_size']}"
    train_data = np.load(data_dir / "train_sequences.npy", mmap_mode='r')
    val_data = np.load(data_dir / "val_sequences.npy", mmap_mode='r')
    
    print(f"📂 Data loaded:")
    print(f"  Train: {{len(train_data):,}} sequences")
    print(f"  Val: {{len(val_data):,}} sequences")
    print()
    
    # Create model
    model = GPTModel(config_obj).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"🏗️  Model created:")
    print(f"  Total parameters: {{total_params:,}}")
    print(f"  Model size: {{total_params * 4 / 1e9:.2f}} GB (FP32)")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        if step > max_steps:
            return min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    
    scaler = GradScaler()
    
    # Setup directories
    checkpoint_dir = PATHS.CHECKPOINTS / "{stage}" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = PATHS.LOGS / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "{stage}_training_log.jsonl"
    
    print(f"📁 Outputs:")
    print(f"  Checkpoints: {{checkpoint_dir}}")
    print(f"  Logs: {{log_file}}")
    print()
    
    print("="*80)
    print("🚀 TRAINING STARTED")
    print("="*80 + "\\n")
    
    # Training state
    model.train()
    best_val_loss = float('inf')
    best_perplexity = float('inf')
    step = 0
    epoch = 0
    
    # Time tracking
    training_start_time = time.time()
    step_times = []
    
    # Main training loop
    while step < max_steps:
        epoch += 1
        indices = np.random.permutation(len(train_data))
        
        for batch_start in range(0, len(indices), batch_size):
            if step >= max_steps:
                break
            
            step_start_time = time.time()
            
            # Get batch
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_data = train_data[batch_indices]
            inputs = torch.tensor(batch_data, dtype=torch.long, device=device)
            targets = inputs.clone()
            
            # Forward pass with mixed precision
            with autocast():
                logits, loss = model(inputs, targets)
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update learning rate
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            step += 1
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            
            # Calculate time metrics
            elapsed_time = time.time() - training_start_time
            avg_step_time = np.mean(step_times[-100:]) if len(step_times) >= 10 else np.mean(step_times)
            steps_per_sec = 1 / avg_step_time if avg_step_time > 0 else 0
            remaining_steps = max_steps - step
            eta_seconds = remaining_steps * avg_step_time
            
            # Logging every 100 steps
            if step % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                current_loss = loss.item() * gradient_accumulation_steps
                
                print(f"Step {{step:6d}} | "
                      f"Loss: {{current_loss:.4f}} | "
                      f"LR: {{lr:.2e}} | "
                      f"Steps/sec: {{steps_per_sec:.2f}} | "
                      f"Elapsed: {{format_time(elapsed_time)}} | "
                      f"ETA: {{format_time(eta_seconds)}}")
                
                # Log to JSONL file
                log_entry = {{
                    'step': step,
                    'epoch': epoch,
                    'loss': current_loss,
                    'lr': lr,
                    'step_time_ms': step_time * 1000,
                    'steps_per_sec': steps_per_sec,
                    'elapsed_seconds': elapsed_time,
                    'eta_seconds': eta_seconds,
                    'timestamp': datetime.now().isoformat(),
                }}
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\\n')
            
            # Evaluation every 1000 steps
            if step % eval_interval == 0:
                val_loss, perplexity = evaluate_with_perplexity(model, val_data, device)
                
                print("\\n" + "="*80)
                print(f"📊 EVALUATION at Step {{step}}")
                print(f"Validation Loss: {{val_loss:.4f}}")
                print(f"Perplexity: {{perplexity:.2f}}")
                print(f"Elapsed: {{format_time(elapsed_time)}} | ETA: {{format_time(eta_seconds)}}")
                
                # Check if best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_perplexity = perplexity
                    print("⭐ New best validation loss!")
                    
                    # Save best model
                    best_model_path = checkpoint_dir / "best_model.pt"
                    save_checkpoint(model, optimizer, step, val_loss, perplexity, config_obj, best_model_path)
                    print("⭐ Saved best model!")
                
                print(f"Best so far: Loss {{best_val_loss:.4f}}, Perplexity {{best_perplexity:.2f}}")
                print("="*80 + "\\n")
                
                # Log evaluation to JSONL
                eval_entry = {{
                    'step': step,
                    'type': 'evaluation',
                    'val_loss': val_loss,
                    'perplexity': perplexity,
                    'best_val_loss': best_val_loss,
                    'best_perplexity': best_perplexity,
                    'elapsed_seconds': elapsed_time,
                    'eta_seconds': eta_seconds,
                    'timestamp': datetime.now().isoformat(),
                }}
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps(eval_entry) + '\\n')
                
                model.train()
            
            # Save checkpoint every 5000 steps
            if step % save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{{step}}.pt"
                save_checkpoint(model, optimizer, step, None, None, config_obj, checkpoint_path)
                print(f"💾 Saved checkpoint: checkpoint_step_{{step}}.pt")
    
    # Training complete
    total_time = time.time() - training_start_time
    
    print("\\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Total steps: {{max_steps:,}}")
    print(f"Best val loss: {{best_val_loss:.4f}}")
    print(f"Best perplexity: {{best_perplexity:.2f}}")
    print(f"Total time: {{format_time(total_time)}}")
    print(f"Average speed: {{max_steps/total_time:.2f}} steps/sec")
    print(f"\\nModel saved to: {{checkpoint_dir}}")
    print()


def evaluate_with_perplexity(model, val_data, device, num_samples=1000):
    """
    Evaluate model and calculate perplexity.
    
    Perplexity = exp(loss)
    Lower perplexity = better model
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Random sample
    indices = np.random.choice(len(val_data), min(num_samples, len(val_data)), replace=False)
    
    with torch.no_grad():
        for i in range(0, len(indices), 4):
            batch_indices = indices[i:i+4]
            batch_data = val_data[batch_indices]
            
            inputs = torch.tensor(batch_data, dtype=torch.long, device=device)
            targets = inputs.clone()
            
            _, loss = model(inputs, targets)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, step, loss, perplexity, config, path):
    """Save model checkpoint with all metadata."""
    checkpoint = {{
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'perplexity': perplexity,
        'best_val_loss': loss,
        'config': {{
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layers,
            'n_head': config.n_heads,
            'n_embd': config.d_model,
            'block_size': config.max_seq_length,
        }},
        'timestamp': datetime.now().isoformat(),
    }}
    torch.save(checkpoint, path, _use_new_zipfile_serialization=True)


if __name__ == "__main__":
    train_model()
'''


# Simple tokenizer and dataset scripts (same as before but with better messages)

def get_tokenizer_script(stage, config):
    return f'''"""
{stage.upper()} - Step 1: Train {config['vocab_size']:,} Token Tokenizer
===============================================================================
Time: ~15-25 minutes
Expected output: tokenizer_vocab{config['vocab_size']}.json
"""

import sys
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tqdm import tqdm
import json

PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))
from paths import PATHS

def train_tokenizer():
    print("\\n" + "="*80)
    print("{stage.upper()} - TRAIN {config['vocab_size']:,} TOKENIZER")
    print("="*80 + "\\n")
    
    vocab_size = {config['vocab_size']}
    
    print(f"Configuration:")
    print(f"  Vocabulary size: {{vocab_size:,}}")
    print(f"  Algorithm: Byte-Pair Encoding (BPE)")
    print(f"  Special tokens: <|endoftext|>, <|pad|>, <|unk|>")
    print()
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
        show_progress=True,
    )
    
    # Load data
    train_file = PATHS.PROCESSED_DATA / "consolidated" / "train.jsonl"
    
    if not train_file.exists():
        print(f"❌ Training file not found: {{train_file}}")
        return
    
    print(f"📂 Training data: {{train_file}}")
    print(f"   Using 100,000 articles")
    print()
    
    def article_iterator():
        count = 0
        max_articles = 100000
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading articles", total=max_articles):
                if count >= max_articles:
                    break
                try:
                    article = json.loads(line.strip())
                    yield article['text']
                    count += 1
                except:
                    continue
    
    print("🔥 Training tokenizer...")
    tokenizer.train_from_iterator(article_iterator(), trainer=trainer, length=100000)
    
    # Save
    output_path = PATHS.TOKENIZER_DATA / f"tokenizer_vocab{{vocab_size}}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    
    print(f"\\n✅ COMPLETE!")
    print(f"Saved to: {{output_path}}")
    print(f"\\n📍 Next step: python scripts/{stage}/step2_tokenize_dataset.py\\n")

if __name__ == "__main__":
    train_tokenizer()
'''


def get_dataset_script(stage, config):
    return f'''"""
{stage.upper()} - Step 2: Tokenize Dataset
===============================================================================
Tokenize 2.88M Wikipedia articles with {config['vocab_size']:,} token vocabulary
Time: ~45-75 minutes
"""

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer

PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))
from paths import PATHS

def tokenize_dataset():
    print("\\n" + "="*80)
    print("{stage.upper()} - TOKENIZE DATASET")
    print("="*80 + "\\n")
    
    # Config
    max_seq_length = 256
    stride = 128
    chunk_size = 10000
    vocab_size = {config['vocab_size']}
    
    # Load tokenizer
    tokenizer_path = PATHS.TOKENIZER_DATA / f"tokenizer_vocab{{vocab_size}}.json"
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {{tokenizer_path}}")
        print("   Please run step1_train_tokenizer.py first!")
        return
    
    print(f"📂 Loading tokenizer: {{tokenizer_path.name}}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    
    print(f"✅ Tokenizer loaded")
    print(f"   Vocab size: {{tokenizer.get_vocab_size():,}}")
    print()
    
    # Output directory
    output_dir = PATHS.TRAINING_DATA / "{stage}_vocab{config['vocab_size']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⚙️  Configuration:")
    print(f"   Max sequence length: {{max_seq_length}}")
    print(f"   Stride: {{stride}}")
    print(f"   Output: {{output_dir}}")
    print()
    
    # Process splits
    for split in ['train', 'val', 'test']:
        input_file = PATHS.PROCESSED_DATA / "consolidated" / f"{{split}}.jsonl"
        if not input_file.exists():
            print(f"⚠️  Skipping {{split}}: file not found")
            continue
        
        print(f"Processing {{split.upper()}}...")
        all_sequences = []
        
        total_articles = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        
        with open(input_file, 'r', encoding='utf-8') as f:
            chunk = []
            for line in tqdm(f, total=total_articles, desc=f"Tokenizing {{split}}"):
                try:
                    article = json.loads(line.strip())
                    chunk.append(article['text'])
                    
                    if len(chunk) >= chunk_size:
                        seqs = process_chunk(chunk, tokenizer, max_seq_length, stride, eos_token_id)
                        all_sequences.extend(seqs)
                        chunk = []
                except:
                    continue
            
            if chunk:
                seqs = process_chunk(chunk, tokenizer, max_seq_length, stride, eos_token_id)
                all_sequences.extend(seqs)
        
        # Save
        sequences_array = np.array(all_sequences, dtype=np.int32)
        output_file = output_dir / f"{{split}}_sequences.npy"
        np.save(output_file, sequences_array)
        
        file_size_gb = output_file.stat().st_size / 1e9
        print(f"✅ {{split.upper()}}: {{len(sequences_array):,}} sequences, {{file_size_gb:.2f}} GB")
    
    print(f"\\n✅ COMPLETE!")
    print(f"Data saved to: {{output_dir}}")
    print(f"\\n📍 Next step: python scripts/{stage}/step3_train_model.py\\n")

def process_chunk(texts, tokenizer, max_seq_length, stride, eos_token_id):
    sequences = []
    for text in texts:
        token_ids = tokenizer.encode(text).ids + [eos_token_id]
        start_idx = 0
        while start_idx < len(token_ids):
            seq = token_ids[start_idx:start_idx + max_seq_length]
            if len(seq) >= 32:
                if len(seq) < max_seq_length:
                    seq += [eos_token_id] * (max_seq_length - len(seq))
                sequences.append(seq)
            start_idx += stride
            if start_idx + 32 > len(token_ids):
                break
    return sequences

if __name__ == "__main__":
    tokenize_dataset()
'''


def get_readme(stage, config):
    return f'''# {stage.upper()} - {config['params']} Parameters

## Overview
- **Parameters**: {config['params']}
- **Architecture**: {config['n_layer']} layers, {config['n_head']} heads, {config['d_model']} d_model
- **Vocabulary**: {config['vocab_size']:,} tokens
- **Training time**: {config['time_train']}
- **Expected loss**: ~{config['loss']}

## Quick Start

```bash
# Step 1: Train tokenizer (~15-25 min)
python scripts/{stage}/step1_train_tokenizer.py

# Step 2: Tokenize dataset (~45-75 min)
python scripts/{stage}/step2_tokenize_dataset.py

# Step 3: Train model ({config['time_train']})
python scripts/{stage}/step3_train_model.py
```

## Enhanced Features

✅ **Time Tracking**
- Elapsed time display
- ETA (time remaining) calculation
- Steps per second

✅ **Perplexity Calculation**
- Computed every 1000 steps
- Logged alongside validation loss
- Tracks best perplexity

✅ **Detailed Logging**
- JSONL log file with all metrics
- Timestamp for each entry
- Both training and evaluation logs

## Monitoring Training

Watch for these metrics:
- **Loss**: Should decrease from ~70 → ~{config['loss']}
- **Perplexity**: Should decrease (lower is better)
- **Steps/sec**: Indicates training speed
- **ETA**: Time remaining

## Output Files

Model checkpoints:
```
E:/WikiForge-GPT/models/checkpoints/{stage}/checkpoints/
├── checkpoint_step_5000.pt
├── checkpoint_step_10000.pt
├── ...
├── checkpoint_step_100000.pt
└── best_model.pt  ⭐ (best validation loss)
```

Training logs:
```
E:/WikiForge-GPT/logs/training/
└── {stage}_training_log.jsonl
```

## Next Steps

After completing this stage, proceed to the next stage folder for continued training!
'''


# =============================================================================
# CREATE ALL FILES
# =============================================================================

def create_all_scripts():
    print(__doc__)
    print("Creating all 12 files with ENHANCED features...")
    print()
    
    created_files = []
    
    for stage, config in CONFIGS.items():
        stage_dir = BASE_DIR / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating {stage.upper()} scripts...")
        
        # Step 1 - Tokenizer
        file1 = stage_dir / "step1_train_tokenizer.py"
        file1.write_text(get_tokenizer_script(stage, config), encoding='utf-8')
        created_files.append(file1)
        
        # Step 2 - Dataset
        file2 = stage_dir / "step2_tokenize_dataset.py"
        file2.write_text(get_dataset_script(stage, config), encoding='utf-8')
        created_files.append(file2)
        
        # Step 3 - Training (IMPROVED with time tracking & perplexity)
        file3 = stage_dir / "step3_train_model.py"
        file3.write_text(get_train_script_improved(stage, config), encoding='utf-8')
        created_files.append(file3)
        
        # README
        file4 = stage_dir / "README.md"
        file4.write_text(get_readme(stage, config), encoding='utf-8')
        created_files.append(file4)
        
        print(f"  ✅ Created 4 files in {stage}")
        print(f"     • Time tracking: ✅")
        print(f"     • ETA calculation: ✅")
        print(f"     • Perplexity: ✅")
        print(f"     • Enhanced logging: ✅")
    
    print()
    print("="*80)
    print("SUCCESS! ALL 12 FILES CREATED WITH ENHANCED FEATURES!")
    print("="*80)
    print()
    print("Enhanced features in ALL training scripts:")
    print("  ✅ Time elapsed tracking")
    print("  ✅ Time remaining (ETA) calculation")
    print("  ✅ Perplexity calculation every 1000 steps")
    print("  ✅ Detailed JSONL logging with timestamps")
    print("  ✅ Progress bars and status updates")
    print()
    print("Files created:")
    for f in created_files:
        print(f"  ✅ {f}")
    
    print()
    print("Next steps:")
    print("  1. Review the created files")
    print("  2. Start with: python scripts/stage_1/step1_train_tokenizer.py")
    print("  3. Monitor logs in E:/WikiForge-GPT/logs/training/")
    print()
    print("Total training time: ~13 days")
    print("Final model: 350M parameters with comprehensive metrics!")
    print()

if __name__ == "__main__":
    create_all_scripts()