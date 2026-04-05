"""
WikiForge-GPT Training Loop (Fixed with Debugging)
===================================================
Train the GPT model from scratch on Wikipedia data.
This version prints immediately and shows what's happening.
"""

import os
import sys
import json
import time
import zipfile
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler  # Updated import
from tqdm import tqdm

# Force immediate output
def print_now(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()

# Import centralized paths
PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))

print_now("Importing paths...")
from paths import PATHS

print_now("Importing GPT architecture...")
from scripts.stage_0.step3_gpt_architecture import GPTModel, GPTConfig

print_now("All imports successful!")


class WikiDataset(Dataset):
    """
    Dataset for loading tokenized Wikipedia data.
    """
    
    def __init__(self, data_path: Path):
        """
        Args:
            data_path: Path to .npy or .npz file with tokenized sequences
        """
        print_now(f"\n📂 Loading dataset from: {data_path}")
        print_now("   This may take 30-60 seconds for large files...")
        
        self.data_path = data_path
        
        # Check file exists
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Check file size
        file_size_gb = data_path.stat().st_size / (1024**3)
        print_now(f"   File size: {file_size_gb:.2f} GB")
        
        # Load data - step2 saves as raw binary memmap files
        print_now("   Loading dataset...")
        
        # Calculate shape from file size
        file_size = data_path.stat().st_size
        bytes_per_element = 4  # int32 = 4 bytes
        seq_length = 256  # From config
        
        # Calculate number of sequences
        total_elements = file_size // bytes_per_element
        num_sequences = total_elements // seq_length
        
        print_now(f"   Calculated shape: ({num_sequences:,}, {seq_length})")
        
        try:
            # Load as raw binary memmap (what step2 actually creates)
            print_now("   Loading as memory-mapped binary file...")
            self.data = np.memmap(
                data_path,
                dtype=np.int32,
                mode='r',
                shape=(num_sequences, seq_length)
            )
            print_now("   ✅ Loaded as memory-mapped file")
        except Exception as e:
            print_now(f"   Memmap failed: {e}")
            raise RuntimeError(f"Could not load dataset from {data_path}")
        
        print_now(f"   Sequences: {len(self.data):,}, Shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Returns a single sequence as torch tensor."""
        sequence = self.data[idx].astype(np.int64)
        return torch.from_numpy(sequence)


class Trainer:
    """Handles GPT model training."""
    
    def __init__(
        self,
        model: GPTModel,
        config: GPTConfig,
        train_dataset: WikiDataset,
        val_dataset: WikiDataset,
        output_dir: Path,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_steps: int = 100000,
        warmup_steps: int = 2000,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        log_interval: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
    ):
        print_now("\n⚙️ Initializing Trainer...")
        
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        
        print_now(f"   Device: {self.device}")
        print_now(f"   Moving model to {self.device}...")
        self.model.to(self.device)
        print_now("   ✅ Model moved to device")
        
        # Create output directories
        self.checkpoint_dir = output_dir / "checkpoints"
        self.samples_dir = output_dir / "samples"
        self.logs_dir = output_dir / "logs"
        
        print_now("   Creating output directories...")
        for d in [self.checkpoint_dir, self.samples_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        print_now("   ✅ Directories created")
        
        # Setup optimizer
        print_now("   Creating optimizer...")
        self.optimizer = self._create_optimizer()
        print_now("   ✅ Optimizer created")
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # Create DataLoaders
        print_now("   Creating DataLoaders...")
        print_now("   NOTE: This may take 1-2 minutes for the first batch...")
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True if device == "cuda" else False,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == "cuda" else False,
        )
        
        print_now("   ✅ DataLoaders created")
        
        self.train_losses = []
        self.val_losses = []
        
        print_now(f"\n{'='*80}")
        print_now("✅ Trainer Initialized Successfully")
        print_now(f"{'='*80}")
        print_now(f"Batch Size: {self.batch_size}")
        print_now(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        print_now(f"Effective Batch Size: {self.batch_size * self.gradient_accumulation_steps}")
        print_now(f"Learning Rate: {self.learning_rate}")
        print_now(f"Max Steps: {self.max_steps:,}")
        print_now(f"{'='*80}\n")
    
    def _create_optimizer(self):
        """Create AdamW optimizer."""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'LayerNorm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        return optimizer
    
    def _get_lr(self, step: int) -> float:
        """Get learning rate with warmup and cosine decay."""
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.learning_rate * 0.5 * (1.0 + np.cos(np.pi * progress))
    
    def _update_lr(self, step: int):
        """Update learning rate."""
        lr = self._get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""
        batch = batch.to(self.device)
        
        with autocast(device_type='cuda', enabled=self.use_amp):
            logits, loss = self.model(batch, labels=batch)
        
        loss = loss / self.gradient_accumulation_steps
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    @torch.no_grad()
    def evaluate(self, max_batches: int = 100) -> float:
        """Evaluate on validation set."""
        print_now(f"\n📊 Evaluating on validation set...")
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            if num_batches >= max_batches:
                break
            
            batch = batch.to(self.device)
            
            with autocast(device_type='cuda', enabled=self.use_amp):
                logits, loss = self.model(batch, labels=batch)
            
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print_now(f"💾 Saved checkpoint: {checkpoint_path.name}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print_now(f"⭐ Saved best model!")
    
    def train(self):
        """Main training loop."""
        
        print_now(f"\n{'='*80}")
        print_now("🚀 Starting Training")
        print_now(f"{'='*80}\n")
        
        print_now("Setting model to training mode...")
        self.model.train()
        
        print_now("Creating training iterator...")
        print_now("NOTE: First batch may take 1-2 minutes to load...\n")
        
        running_loss = 0.0
        train_iter = iter(self.train_loader)
        
        # Training start time
        training_start_time = time.time()
        step_start_time = time.time()
        
        print_now("Starting main training loop...\n")
        
        while self.current_step < self.max_steps:
            
            # Show progress every 10 steps at the start
            if self.current_step < 100 and self.current_step % 10 == 0:
                print_now(f"Step {self.current_step}...")
            
            self.optimizer.zero_grad()
            
            # Gradient accumulation
            for accum_step in range(self.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                
                loss = self.train_step(batch)
                running_loss += loss
            
            # Optimizer step
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            lr = self._update_lr(self.current_step)
            self.current_step += 1
            
            # Calculate step time
            step_end_time = time.time()
            step_time_ms = (step_end_time - step_start_time) * 1000
            step_start_time = step_end_time
            
            # Logging every 100 steps
            if self.current_step % self.log_interval == 0:
                avg_loss = running_loss / self.log_interval
                
                # Calculate timing statistics
                elapsed_time = time.time() - training_start_time
                steps_remaining = self.max_steps - self.current_step
                steps_per_sec = self.log_interval / (elapsed_time if self.current_step == self.log_interval else (time.time() - (training_start_time + elapsed_time - self.log_interval / (self.current_step / elapsed_time))))
                
                # Better calculation
                if self.current_step > 0:
                    avg_time_per_step = elapsed_time / self.current_step
                    eta_seconds = steps_remaining * avg_time_per_step
                    
                    # Format ETA
                    eta_hours = int(eta_seconds // 3600)
                    eta_mins = int((eta_seconds % 3600) // 60)
                    eta_secs = int(eta_seconds % 60)
                    eta_str = f"{eta_hours}h{eta_mins:02d}m{eta_secs:02d}s"
                    
                    # Format elapsed time
                    elapsed_hours = int(elapsed_time // 3600)
                    elapsed_mins = int((elapsed_time % 3600) // 60)
                    elapsed_secs = int(elapsed_time % 60)
                    elapsed_str = f"{elapsed_hours}h{elapsed_mins:02d}m{elapsed_secs:02d}s"
                    
                    # Get VRAM usage
                    if torch.cuda.is_available():
                        vram_used = torch.cuda.memory_allocated() / 1024**3
                        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        vram_str = f"{vram_used:.1f}/{vram_total:.1f}GB"
                    else:
                        vram_str = "N/A"
                    
                    # Progress percentage
                    progress = (self.current_step / self.max_steps) * 100
                    
                    print_now(
                        f"Step {self.current_step:>6}/{self.max_steps} ({progress:>5.1f}%) | "
                        f"Loss: {avg_loss:>6.4f} | LR: {lr:.2e} | "
                        f"{step_time_ms:>5.1f}ms/step | "
                        f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                        f"VRAM: {vram_str}"
                    )
                
                # Save to log
                log_entry = {
                    'step': self.current_step,
                    'loss': avg_loss,
                    'lr': lr,
                    'step_time_ms': step_time_ms,
                    'elapsed_time': elapsed_time if self.current_step > 0 else 0,
                    'eta_seconds': eta_seconds if self.current_step > 0 else 0,
                }
                
                log_file = self.logs_dir / "training_log.jsonl"
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                
                running_loss = 0.0
            
            # Evaluation every 1000 steps
            if self.current_step % self.eval_interval == 0:
                val_loss = self.evaluate()
                print_now(f"\n{'='*80}")
                print_now(f"📊 EVALUATION at Step {self.current_step}")
                print_now(f"Validation Loss: {val_loss:.4f}")
                
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print_now(f"⭐ New best validation loss!")
                
                print_now(f"{'='*80}\n")
                
                self.save_checkpoint(self.current_step, is_best=is_best)
            
            # Regular checkpointing
            elif self.current_step % self.save_interval == 0:
                self.save_checkpoint(self.current_step)
        
        # Training complete
        total_time = time.time() - training_start_time
        total_hours = int(total_time // 3600)
        total_mins = int((total_time % 3600) // 60)
        total_secs = int(total_time % 60)
        
        print_now(f"\n{'='*80}")
        print_now("✅ TRAINING COMPLETE!")
        print_now(f"{'='*80}")
        print_now(f"Total steps: {self.current_step}")
        print_now(f"Best val loss: {self.best_val_loss:.4f}")
        print_now(f"Total time: {total_hours}h{total_mins:02d}m{total_secs:02d}s")
        print_now(f"Average: {total_time/self.current_step:.2f}s per step")


def main():
    """Main execution."""
    
    print_now("="*80)
    print_now("WikiForge-GPT Training - Stage 0 (Tiny Model)")
    print_now("="*80)
    
    # Paths
    data_dir = PATHS.TRAINING_DATA / "stage_0_tiny_vocab8000"
    output_dir = PATHS.CHECKPOINTS / "stage_0_tiny"
    
    print_now(f"\nData directory: {data_dir}")
    print_now(f"Output directory: {output_dir}")
    
    # Check CUDA
    print_now(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print_now(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print_now(f"CUDA version: {torch.version.cuda}")
    
    # Load datasets
    print_now("\n" + "="*80)
    print_now("STEP 1: Loading Datasets")
    print_now("="*80)
    
    try:
        train_dataset = WikiDataset(data_dir / "train_tokenized.npy")
        val_dataset = WikiDataset(data_dir / "val_tokenized.npy")
        print_now("✅ Datasets loaded successfully!")
    except Exception as e:
        print_now(f"❌ ERROR loading datasets: {e}")
        raise
    
    # Create model
    print_now("\n" + "="*80)
    print_now("STEP 2: Creating Model")
    print_now("="*80)
    
    try:
        config = GPTConfig(
            vocab_size=8000,
            n_layers=4,
            n_heads=4,
            d_model=256,
            d_ff=1024,
            max_seq_length=256,
            dropout=0.1,
        )
        
        model = GPTModel(config)
        print_now("✅ Model created successfully!")
    except Exception as e:
        print_now(f"❌ ERROR creating model: {e}")
        raise
    
    # Create trainer
    print_now("\n" + "="*80)
    print_now("STEP 3: Creating Trainer")
    print_now("="*80)
    
    try:
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=3e-4,
            max_steps=100000,
            warmup_steps=2000,
            eval_interval=1000,
            save_interval=5000,
        )
        print_now("✅ Trainer created successfully!")
    except Exception as e:
        print_now(f"❌ ERROR creating trainer: {e}")
        raise
    
    # Start training
    print_now("\n" + "="*80)
    print_now("STEP 4: Training")
    print_now("="*80)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print_now("\n\n⚠️ Training interrupted by user")
        print_now("Progress has been saved in checkpoints")
    except Exception as e:
        print_now(f"\n\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()