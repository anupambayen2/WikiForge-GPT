"""
STAGE_1 - Step 3: Train 45M Model (FIXED DATA LOADING)
===============================================================================
Architecture: 8 layers, 8 heads, 512 d_model
Expected time: ~14 hours
Expected loss: ~2.5

Uses correct data loading from Stage 0!
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
from torch.amp import autocast, GradScaler
from datetime import datetime, timedelta

PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))
from paths import PATHS
from scripts.step3_gpt_architecture import GPTModel, GPTConfig


def format_time(seconds):
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
    else:
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days}d {hours}h"


def load_data(data_path, seq_length=256):
    """
    Load data using same method as Stage 0 (memory-mapped binary).
    """
    print(f"📂 Loading dataset from: {data_path}")
    
    # Calculate shape from file size
    file_size = data_path.stat().st_size
    bytes_per_element = 4  # int32 = 4 bytes
    total_elements = file_size // bytes_per_element
    num_sequences = total_elements // seq_length
    
    print(f"   File size: {file_size / 1e9:.2f} GB")
    print(f"   Calculated shape: ({num_sequences:,}, {seq_length})")
    
    # Load as memory-mapped binary file (what step2 creates)
    data = np.memmap(
        data_path,
        dtype=np.int32,
        mode='r',
        shape=(num_sequences, seq_length)
    )
    
    print(f"   ✅ Loaded: {len(data):,} sequences")
    return data


def train_model():
    print("\n" + "="*80)
    print("STAGE_1 - TRAIN 45M MODEL")
    print("="*80 + "\n")
    
    # Configuration
    config_obj = GPTConfig(
        vocab_size=16000,
        n_layers=8,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        max_seq_length=256,
        dropout=0.1,
        pad_token_id=1,
        eos_token_id=0,
    )
    
    batch_size = 2
    gradient_accumulation_steps = 16
    learning_rate = 3e-4
    max_steps = 100000
    warmup_steps = 2000
    min_lr = 3e-5
    
    log_interval = 100
    eval_interval = 1000
    save_interval = 5000
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📋 Configuration:")
    print(f"  Model: 45M parameters")
    print(f"  Architecture: 8 layers, 8 heads, 512 d_model")
    print(f"  Vocab size: 16,000")
    print(f"  Batch size: {batch_size} × {gradient_accumulation_steps} = {batch_size*gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate} → {min_lr}")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Device: {device}")
    print()
    
    # Load data using correct method
    data_dir = PATHS.TRAINING_DATA / "stage_1_vocab16000"
    
    train_data = load_data(data_dir / "train_sequences.npy")
    val_data = load_data(data_dir / "val_sequences.npy")
    
    print()
    
    # Create model
    model = GPTModel(config_obj).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"🏗️  Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1e9:.2f} GB (FP32)")
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
    
    scaler = GradScaler('cuda')
    
    # Setup directories
    checkpoint_dir = PATHS.CHECKPOINTS / "stage_1" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = PATHS.LOGS / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "stage_1_training_log.jsonl"
    
    print(f"📁 Outputs:")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {log_file}")
    print()
    
    print("="*80)
    print("🚀 TRAINING STARTED")
    print("="*80 + "\n")
    
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
            with autocast('cuda'):
                logits, loss = model(inputs, labels=targets)
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
                
                print(f"Step {step:6d} | "
                      f"Loss: {current_loss:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Steps/sec: {steps_per_sec:.2f} | "
                      f"Elapsed: {format_time(elapsed_time)} | "
                      f"ETA: {format_time(eta_seconds)}")
                
                # Log to JSONL file
                log_entry = {
                    'step': step,
                    'epoch': epoch,
                    'loss': current_loss,
                    'lr': lr,
                    'step_time_ms': step_time * 1000,
                    'steps_per_sec': steps_per_sec,
                    'elapsed_seconds': elapsed_time,
                    'eta_seconds': eta_seconds,
                    'timestamp': datetime.now().isoformat(),
                }
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            
            # Evaluation every 1000 steps
            if step % eval_interval == 0:
                val_loss, perplexity = evaluate_with_perplexity(model, val_data, device)
                
                print("\n" + "="*80)
                print(f"📊 EVALUATION at Step {step}")
                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Perplexity: {perplexity:.2f}")
                print(f"Elapsed: {format_time(elapsed_time)} | ETA: {format_time(eta_seconds)}")
                
                # Check if best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_perplexity = perplexity
                    print("⭐ New best validation loss!")
                    
                    # Save best model
                    best_model_path = checkpoint_dir / "best_model.pt"
                    save_checkpoint(model, optimizer, step, val_loss, perplexity, config_obj, best_model_path)
                    print("⭐ Saved best model!")
                
                print(f"Best so far: Loss {best_val_loss:.4f}, Perplexity {best_perplexity:.2f}")
                print("="*80 + "\n")
                
                # Log evaluation to JSONL
                eval_entry = {
                    'step': step,
                    'type': 'evaluation',
                    'val_loss': val_loss,
                    'perplexity': perplexity,
                    'best_val_loss': best_val_loss,
                    'best_perplexity': best_perplexity,
                    'elapsed_seconds': elapsed_time,
                    'eta_seconds': eta_seconds,
                    'timestamp': datetime.now().isoformat(),
                }
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps(eval_entry) + '\n')
                
                model.train()
            
            # Save checkpoint every 5000 steps
            if step % save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
                save_checkpoint(model, optimizer, step, None, None, config_obj, checkpoint_path)
                print(f"💾 Saved checkpoint: checkpoint_step_{step}.pt")
    
    # Training complete
    total_time = time.time() - training_start_time
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Total steps: {max_steps:,}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best perplexity: {best_perplexity:.2f}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Average speed: {max_steps/total_time:.2f} steps/sec")
    print(f"\nModel saved to: {checkpoint_dir}")
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
            
            _, loss = model(inputs, labels=targets)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, step, loss, perplexity, config, path):
    """Save model checkpoint with all metadata."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'perplexity': perplexity,
        'best_val_loss': loss,
        'config': {
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layers,
            'n_head': config.n_heads,
            'n_embd': config.d_model,
            'block_size': config.max_seq_length,
        },
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(checkpoint, path, _use_new_zipfile_serialization=True)


if __name__ == "__main__":
    train_model()