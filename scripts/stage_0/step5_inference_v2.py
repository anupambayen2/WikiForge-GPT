"""
WikiForge-GPT Inference v2 - Enhanced Anti-Repetition
======================================================
Improved text generation with frequency-based penalties.

Usage:
    python scripts/step5_inference_v2.py --interactive
"""

import sys
import torch
import json
from pathlib import Path
from typing import List, Optional, Dict
import argparse
from collections import defaultdict

# Import centralized paths
PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))

from paths import PATHS
from tokenizers import Tokenizer
from scripts.stage_0.step3_gpt_architecture import GPTModel, GPTConfig


class TextGeneratorV2:
    """
    Enhanced text generator with advanced anti-repetition.
    """
    
    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        print(f"\n{'='*80}")
        print("WikiForge-GPT Text Generator v2 (Enhanced)")
        print(f"{'='*80}\n")
        
        # Load tokenizer
        print(f"📂 Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"✅ Tokenizer loaded (vocab size: {self.vocab_size:,})")
        
        # Get special tokens
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        
        # Load model
        print(f"\n📂 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get config from checkpoint
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            self.config = GPTConfig(
                vocab_size=config_dict.get('vocab_size', 8000),
                n_layers=config_dict.get('n_layer', 4),
                n_heads=config_dict.get('n_head', 4),
                d_model=config_dict.get('n_embd', 256),
                d_ff=config_dict.get('n_embd', 256) * 4,
                max_seq_length=config_dict.get('block_size', 256),
                dropout=0.1,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
        else:
            # Default Stage 0 config
            self.config = GPTConfig(
                vocab_size=8000,
                n_layers=4,
                n_heads=4,
                d_model=256,
                d_ff=1024,
                max_seq_length=256,
                dropout=0.1,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
        
        # Create model
        self.model = GPTModel(self.config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])
        
        self.model.to(device)
        self.model.eval()
        
        # Print model info
        step = checkpoint.get('step', 'unknown')
        val_loss = checkpoint.get('loss', checkpoint.get('best_val_loss', 'unknown'))
        
        print(f"✅ Model loaded successfully!")
        print(f"\n📊 Model Information:")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Training step: {step}")
        print(f"   Validation loss: {val_loss}")
        print(f"   Device: {device}")
        print(f"\n{'='*80}\n")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,  # Higher default
        top_k: Optional[int] = 50,  # More diverse
        top_p: Optional[float] = 0.95,  # Broader sampling
        repetition_penalty: float = 1.8,  # Much stronger!
        frequency_penalty: float = 0.5,  # NEW: penalize frequent tokens
        presence_penalty: float = 0.3,  # NEW: penalize seen tokens
        stop_on_eos: bool = True,
        no_repeat_ngram_size: int = 3,  # NEW: prevent n-gram repetition
    ) -> str:
        """
        Generate text with advanced anti-repetition mechanisms.
        
        Args:
            prompt: Initial text
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            repetition_penalty: Standard repetition penalty (1.8 is strong)
            frequency_penalty: Reduce probability of frequent tokens (0.0-1.0)
            presence_penalty: Reduce probability of any seen token (0.0-1.0)
            stop_on_eos: Stop at EOS token
            no_repeat_ngram_size: Prevent repeating n-grams (3 = trigrams)
        
        Returns:
            Generated text
        """
        
        # Encode prompt
        input_ids = self.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Track token frequencies for advanced penalties
        token_counts: Dict[int, int] = defaultdict(int)
        for token_id in input_ids:
            token_counts[token_id] += 1
        
        # Track n-grams to prevent repetition
        generated_ngrams = set()
        
        # Generate
        for step in range(max_new_tokens):
            # Get predictions
            logits, _ = self.model(input_tensor)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply standard repetition penalty
            if repetition_penalty != 1.0:
                for token_id, count in token_counts.items():
                    # Stronger penalty for more frequent tokens
                    penalty = repetition_penalty ** count
                    next_token_logits[token_id] /= penalty
            
            # Apply frequency penalty (linear with frequency)
            if frequency_penalty > 0:
                for token_id, count in token_counts.items():
                    next_token_logits[token_id] -= frequency_penalty * count
            
            # Apply presence penalty (binary - was token seen?)
            if presence_penalty > 0:
                for token_id in token_counts.keys():
                    next_token_logits[token_id] -= presence_penalty
            
            # Prevent n-gram repetition
            if no_repeat_ngram_size > 0 and len(input_tensor[0]) >= no_repeat_ngram_size:
                # Get last n-1 tokens
                ngram_prefix = tuple(input_tensor[0, -(no_repeat_ngram_size-1):].tolist())
                
                # Block tokens that would create repeated n-grams
                for token_id in range(self.vocab_size):
                    ngram = ngram_prefix + (token_id,)
                    if ngram in generated_ngrams:
                        next_token_logits[token_id] = float('-inf')
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update token counts
            token_counts[next_token.item()] += 1
            
            # Update n-grams
            if no_repeat_ngram_size > 0 and len(input_tensor[0]) >= no_repeat_ngram_size - 1:
                ngram_prefix = tuple(input_tensor[0, -(no_repeat_ngram_size-1):].tolist())
                ngram = ngram_prefix + (next_token.item(),)
                generated_ngrams.add(ngram)
            
            # Append to sequence
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS
            if stop_on_eos and next_token.item() == self.eos_token_id:
                break
            
            # Prevent exceeding max sequence length
            if input_tensor.size(1) >= self.config.max_seq_length:
                break
        
        # Decode
        generated_ids = input_tensor[0].tolist()
        generated_text = self.decode(generated_ids)
        
        return generated_text
    
    def interactive_mode(self):
        """Interactive generation with presets."""
        
        print("\n" + "="*80)
        print("🎮 INTERACTIVE MODE (Enhanced)")
        print("="*80)
        print("\nPresets:")
        print("  1. 'balanced' - Good mix of coherence and creativity")
        print("  2. 'creative' - More diverse, less repetitive")
        print("  3. 'focused' - More coherent, conservative")
        print("\nCommands:")
        print("  - <prompt> - Generate text")
        print("  - 'preset balanced/creative/focused' - Switch preset")
        print("  - 'settings' - Show current settings")
        print("  - 'quit' - Exit")
        print("="*80 + "\n")
        
        # Presets
        presets = {
            'balanced': {
                'max_new_tokens': 100,
                'temperature': 1.0,
                'top_k': 50,
                'top_p': 0.95,
                'repetition_penalty': 1.8,
                'frequency_penalty': 0.5,
                'presence_penalty': 0.3,
                'no_repeat_ngram_size': 3,
            },
            'creative': {
                'max_new_tokens': 100,
                'temperature': 1.2,
                'top_k': 60,
                'top_p': 0.98,
                'repetition_penalty': 2.0,
                'frequency_penalty': 0.7,
                'presence_penalty': 0.5,
                'no_repeat_ngram_size': 4,
            },
            'focused': {
                'max_new_tokens': 100,
                'temperature': 0.7,
                'top_k': 30,
                'top_p': 0.9,
                'repetition_penalty': 1.5,
                'frequency_penalty': 0.3,
                'presence_penalty': 0.2,
                'no_repeat_ngram_size': 2,
            },
        }
        
        current_preset = 'balanced'
        settings = presets[current_preset].copy()
        
        print(f"Current preset: {current_preset}")
        
        while True:
            try:
                prompt = input("\n📝 Prompt: ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if prompt.lower().startswith('preset '):
                    preset_name = prompt.split(' ', 1)[1].strip()
                    if preset_name in presets:
                        current_preset = preset_name
                        settings = presets[current_preset].copy()
                        print(f"✅ Switched to '{current_preset}' preset")
                    else:
                        print(f"❌ Unknown preset. Available: {', '.join(presets.keys())}")
                    continue
                
                if prompt.lower() == 'settings':
                    print(f"\n⚙️ Current Settings (preset: {current_preset}):")
                    for key, value in settings.items():
                        print(f"   {key}: {value}")
                    continue
                
                # Generate
                print(f"\n🤖 Generating...")
                print("-" * 80)
                
                generated = self.generate(prompt=prompt, **settings)
                
                print(generated)
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue


def main():
    """Main execution."""
    
    parser = argparse.ArgumentParser(description='Enhanced WikiForge-GPT Text Generation')
    parser.add_argument('--model', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--tokenizer', type=str, default=None, help='Tokenizer path')
    parser.add_argument('--prompt', type=str, default=None, help='Generation prompt')
    parser.add_argument('--preset', type=str, default='balanced', 
                        choices=['balanced', 'creative', 'focused'],
                        help='Generation preset')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Default paths
    if args.model is None:
        model_path = PATHS.CHECKPOINTS / "stage_0_tiny" / "checkpoints" / "best_model.pt"
    else:
        model_path = Path(args.model)
    
    if args.tokenizer is None:
        tokenizer_path = PATHS.TOKENIZER_DATA / "tokenizer_vocab8000.json"
    else:
        tokenizer_path = Path(args.tokenizer)
    
    # Check files exist
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return
    
    # Create generator
    generator = TextGeneratorV2(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    
    # Interactive mode
    if args.interactive or args.prompt is None:
        generator.interactive_mode()
        return
    
    # Single prompt with preset
    presets = {
        'balanced': {
            'max_new_tokens': 100,
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.8,
            'frequency_penalty': 0.5,
            'presence_penalty': 0.3,
            'no_repeat_ngram_size': 3,
        },
        'creative': {
            'max_new_tokens': 100,
            'temperature': 1.2,
            'top_k': 60,
            'top_p': 0.98,
            'repetition_penalty': 2.0,
            'frequency_penalty': 0.7,
            'presence_penalty': 0.5,
            'no_repeat_ngram_size': 4,
        },
        'focused': {
            'max_new_tokens': 100,
            'temperature': 0.7,
            'top_k': 30,
            'top_p': 0.9,
            'repetition_penalty': 1.5,
            'frequency_penalty': 0.3,
            'presence_penalty': 0.2,
            'no_repeat_ngram_size': 2,
        },
    }
    
    print(f"\n📝 Prompt: {args.prompt}")
    print(f"⚙️ Preset: {args.preset}\n")
    print("="*80)
    
    generated = generator.generate(prompt=args.prompt, **presets[args.preset])
    
    print(generated)
    print("="*80)


if __name__ == "__main__":
    main()