"""
WikiForge-GPT Inference & Text Generation
==========================================
Test your trained GPT model and generate text samples.

Usage:
    python scripts/step5_inference.py
    
    Or with custom prompt:
    python scripts/step5_inference.py --prompt "The history of artificial intelligence"
"""

import sys
import torch
import json
from pathlib import Path
from typing import List, Optional
import argparse

# Import centralized paths
PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))

from paths import PATHS
from tokenizers import Tokenizer
from scripts.stage_0.step3_gpt_architecture import GPTModel, GPTConfig


class TextGenerator:
    """
    Generate text using trained GPT model.
    Supports multiple generation strategies.
    """
    
    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        print(f"\n{'='*80}")
        print("WikiForge-GPT Text Generator")
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
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.2,
        stop_on_eos: bool = True,
    ) -> str:
        """
        Generate text continuation from prompt.
        
        Args:
            prompt: Initial text to continue from
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            repetition_penalty: Penalty for repeating tokens
            stop_on_eos: Stop generation at EOS token
        
        Returns:
            Generated text (prompt + continuation)
        """
        
        # Encode prompt
        input_ids = self.encode(prompt)
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Track generated tokens for repetition penalty
        generated_tokens = set(input_ids)
        
        # Generate
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self.model(input_tensor)
            
            # Get last token's logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated_tokens:
                    next_token_logits[token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated tokens
            generated_tokens.add(next_token.item())
            
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
        """Interactive text generation mode."""
        
        print("\n" + "="*80)
        print("🎮 INTERACTIVE MODE")
        print("="*80)
        print("\nEnter prompts to generate text. Type 'quit' or 'exit' to stop.")
        print("Commands:")
        print("  - Type your prompt and press Enter")
        print("  - 'settings' - Show/modify generation settings")
        print("  - 'examples' - Show example prompts")
        print("  - 'save' - Save last generation")
        print("  - 'quit' - Exit")
        print("="*80 + "\n")
        
        # Default settings
        settings = {
            'max_new_tokens': 100,
            'temperature': 0.8,
            'top_k': 40,
            'top_p': 0.9,
            'repetition_penalty': 1.2,
        }
        
        last_generation = None
        
        while True:
            try:
                prompt = input("\n📝 Prompt: ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if prompt.lower() == 'settings':
                    print("\n⚙️ Current Settings:")
                    for key, value in settings.items():
                        print(f"   {key}: {value}")
                    print("\nTo change: type setting_name=value (e.g., temperature=1.0)")
                    continue
                
                if prompt.lower() == 'examples':
                    self.show_examples()
                    continue
                
                if prompt.lower() == 'save' and last_generation:
                    self.save_generation(last_generation, prompt)
                    continue
                
                # Check for setting changes
                if '=' in prompt:
                    try:
                        key, value = prompt.split('=')
                        key = key.strip()
                        if key in settings:
                            settings[key] = type(settings[key])(value.strip())
                            print(f"✅ Updated {key} to {settings[key]}")
                        continue
                    except:
                        pass
                
                # Generate
                print(f"\n🤖 Generating (max {settings['max_new_tokens']} tokens)...")
                print("-" * 80)
                
                generated = self.generate(
                    prompt=prompt,
                    **settings
                )
                
                print(generated)
                print("-" * 80)
                
                # Show token count
                tokens = self.encode(generated)
                print(f"\n📊 Generated {len(tokens) - len(self.encode(prompt))} new tokens")
                
                last_generation = generated
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    def show_examples(self):
        """Show example prompts."""
        
        examples = [
            "The history of artificial intelligence",
            "In the year 2050,",
            "The most important discovery in science was",
            "Wikipedia is an online encyclopedia that",
            "Machine learning algorithms can",
            "The Python programming language",
            "During World War II,",
            "The universe began with",
        ]
        
        print("\n💡 Example Prompts:")
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")
    
    def save_generation(self, text: str, prompt: str):
        """Save generated text to file."""
        
        output_dir = PATHS.SAMPLES
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"generation_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write("="*80 + "\n\n")
            f.write(text)
        
        print(f"\n💾 Saved to: {filename}")
    
    def batch_generate(self, prompts: List[str], **kwargs):
        """Generate text for multiple prompts."""
        
        print(f"\n🚀 Batch Generation ({len(prompts)} prompts)\n")
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*80}")
            print(f"Prompt {i}/{len(prompts)}: {prompt}")
            print('='*80)
            
            generated = self.generate(prompt, **kwargs)
            print(generated)
            
            results.append({
                'prompt': prompt,
                'generated': generated,
            })
        
        return results


def main():
    """Main execution."""
    
    parser = argparse.ArgumentParser(description='Generate text with WikiForge-GPT')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model checkpoint (default: best_model.pt)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default=None,
        help='Path to tokenizer (default: tokenizer_vocab8000.json)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum tokens to generate (default: 100)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='Top-k sampling (default: 40)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p (nucleus) sampling (default: 0.9)'
    )
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Run with example prompts'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode'
    )
    
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
        print(f"\nAvailable checkpoints:")
        checkpoint_dir = PATHS.CHECKPOINTS / "stage_0_tiny" / "checkpoints"
        if checkpoint_dir.exists():
            for f in sorted(checkpoint_dir.glob("*.pt")):
                print(f"   - {f.name}")
        return
    
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return
    
    # Create generator
    generator = TextGenerator(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    
    # Interactive mode
    if args.interactive or (args.prompt is None and not args.examples):
        generator.interactive_mode()
        return
    
    # Example prompts
    if args.examples:
        example_prompts = [
            "The history of artificial intelligence",
            "In the year 2050,",
            "Wikipedia is an online encyclopedia that",
            "Machine learning algorithms can",
            "The Python programming language",
        ]
        
        generator.batch_generate(
            example_prompts,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        return
    
    # Single prompt
    if args.prompt:
        print(f"\n📝 Prompt: {args.prompt}\n")
        print("🤖 Generating...\n")
        print("="*80)
        
        generated = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        
        print(generated)
        print("="*80)
        
        # Show stats
        tokens = generator.encode(generated)
        prompt_tokens = generator.encode(args.prompt)
        print(f"\n📊 Statistics:")
        print(f"   Prompt tokens: {len(prompt_tokens)}")
        print(f"   Generated tokens: {len(tokens) - len(prompt_tokens)}")
        print(f"   Total tokens: {len(tokens)}")


if __name__ == "__main__":
    main()