"""
Quick Test v3 - Balanced Settings for Small Models
===================================================
Properly calibrated for 5.3M parameter models.

Run: python scripts/quick_test_v3.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage_0.step5_inference_v2 import TextGeneratorV2
from paths import PATHS


def main():
    """Quick test with BALANCED settings."""
    
    print("\n" + "="*80)
    print("🚀 QUICK TEST v3 - Balanced Settings (Calibrated for Small Models)")
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
    
    print("\n🎯 Testing with BALANCED settings (sweet spot for 5.3M params)...\n")
    print("Comparison:")
    print("  v1: Repetitive but coherent")
    print("  v2: Diverse but gibberish")
    print("  v3: BALANCED - coherent AND less repetitive ✨")
    print("\nSettings:")
    print("  • Temperature: 0.9 (slightly more diverse)")
    print("  • Repetition penalty: 1.3 (moderate)")
    print("  • Frequency penalty: 0.2 (light)")
    print("  • N-gram blocking: 3 (prevent exact phrase repetition)")
    print("")
    
    # BALANCED settings - calibrated for small models
    settings = {
        'max_new_tokens': 50,
        'temperature': 0.9,  # Slightly higher than v1, much lower than v2
        'top_k': 40,  # Back to v1 level
        'top_p': 0.92,  # Between v1 and v2
        'repetition_penalty': 1.3,  # Moderate (not too weak, not too strong)
        'frequency_penalty': 0.2,  # Light penalty (not 0.5!)
        'presence_penalty': 0.1,  # Very light (not 0.3!)
        'no_repeat_ngram_size': 3,  # Keep this - prevents exact repetition
    }
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/5")
        print(f"{'='*80}")
        print(f"\n📝 Prompt: {prompt}\n")
        print("🤖 Generated text:")
        print("-" * 80)
        
        generated = generator.generate(prompt=prompt, **settings)
        
        print(generated)
        print("-" * 80)
    
    print(f"\n{'='*80}")
    print("✅ BALANCED TEST COMPLETE!")
    print(f"{'='*80}")
    print("\n💡 This should be the sweet spot!")
    print("   - Less repetitive than v1")
    print("   - Much more coherent than v2")
    print("")
    print("If still too repetitive, try these tweaks:")
    print("   • Increase temperature to 1.0")
    print("   • Increase repetition_penalty to 1.4")
    print("")
    print("If too random/incoherent, try:")
    print("   • Decrease temperature to 0.8")
    print("   • Decrease frequency_penalty to 0.1")
    print("")


if __name__ == "__main__":
    main()