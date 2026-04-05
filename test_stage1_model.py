"""
Quick Test - Stage 1 Model (33.5M parameters)
==============================================
Generate sample text to evaluate quality.
"""

import sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))

from paths import PATHS
from scripts.step3_gpt_architecture import GPTModel, GPTConfig

def test_stage1_model():
    print("\n" + "="*80)
    print("TESTING STAGE 1 MODEL (33.5M parameters)")
    print("="*80 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Load tokenizer
    tokenizer_path = PATHS.TOKENIZER_DATA / "tokenizer_vocab16000.json"
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"✅ Tokenizer loaded (vocab: {tokenizer.get_vocab_size():,})\n")
    
    # Create model config
    config = GPTConfig(
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
    
    # Load model
    model_path = PATHS.CHECKPOINTS / "stage_1" / "checkpoints" / "best_model.pt"
    print(f"Loading model: {model_path}")
    
    model = GPTModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded")
    print(f"   Step: {checkpoint['step']:,}")
    print(f"   Loss: {checkpoint['loss']:.4f}")
    print(f"   Perplexity: {checkpoint['perplexity']:.2f}\n")
    
    print("="*80)
    print("GENERATING TEXT SAMPLES")
    print("="*80 + "\n")
    
    # Test prompts
    test_prompts = [
        "Python is a programming language",
        "The history of artificial intelligence",
        "Machine learning is",
        "Wikipedia is a free",
        "In the year 1991",
    ]
    
    for prompt_text in test_prompts:
        print(f"Prompt: \"{prompt_text}\"")
        print("-" * 80)
        
        # Tokenize prompt
        encoding = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=40
            )
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        print(f"Generated:\n{generated_text}\n")
        print("="*80 + "\n")
    
    print("✅ Testing complete!")
    print("\nModel quality assessment:")
    print("  - Check if text is coherent")
    print("  - Check if it completes sentences logically")
    print("  - Compare to Stage 0 quality")
    print("\nIf quality is good → Proceed to Stage 3!")
    print()

if __name__ == "__main__":
    test_stage1_model()