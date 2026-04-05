"""
Quick Test - Generate text samples immediately
===============================================
Simple script to quickly test your trained model.

Run: python scripts/quick_test.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage_0.step5_inference_v2 import TextGeneratorV2
from paths import PATHS


def main():
    """Quick test with pre-defined prompts."""
    
    print("\n" + "="*80)
    print("🚀 QUICK TEST - WikiForge-GPT")
    print("="*80 + "\n")
    
    # Paths
    model_path = PATHS.CHECKPOINTS / "stage_0_tiny" / "checkpoints" / "best_model.pt"
    tokenizer_path = PATHS.TOKENIZER_DATA / "tokenizer_vocab8000.json"
    
    # Create generator
    generator = TextGeneratorV2(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    
    # Test prompts
    test_prompts = [
        "The history of artificial intelligence",
        "Python is a programming language",
        "In the year 2050,",
        "The theory of relativity",
        "Machine learning is",
    ]
    
    print("\n🎯 Testing with 5 sample prompts...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/5")
        print(f"{'='*80}")
        print(f"\n📝 Prompt: {prompt}\n")
        print("🤖 Generated text:")
        print("-" * 80)
        
        generated = generator.generate(
            prompt=prompt,
            max_new_tokens=50,  # Keep it short for quick test
            temperature=0.8,
            top_k=40,
            top_p=0.9,
        )
        
        print(generated)
        print("-" * 80)
    
    print(f"\n{'='*80}")
    print("✅ QUICK TEST COMPLETE!")
    print(f"{'='*80}")
    print("\n💡 To try your own prompts, run:")
    print("   python scripts/step5_inference.py --interactive")
    print("\nOr with a specific prompt:")
    print('   python scripts/step5_inference.py --prompt "Your prompt here"')
    print("")


if __name__ == "__main__":
    main()