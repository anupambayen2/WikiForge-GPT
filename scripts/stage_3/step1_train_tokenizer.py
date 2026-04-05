"""
STAGE_3 - Step 1: Train 50,000 Token Tokenizer
===============================================================================
Time: ~15-25 minutes
Expected output: tokenizer_vocab50000.json
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
    print("\n" + "="*80)
    print("STAGE_3 - TRAIN 50,000 TOKENIZER")
    print("="*80 + "\n")
    
    vocab_size = 50000
    
    print(f"Configuration:")
    print(f"  Vocabulary size: {vocab_size:,}")
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
        print(f"❌ Training file not found: {train_file}")
        return
    
    print(f"📂 Training data: {train_file}")
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
    output_path = PATHS.TOKENIZER_DATA / f"tokenizer_vocab{vocab_size}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    
    print(f"\n✅ COMPLETE!")
    print(f"Saved to: {output_path}")
    print(f"\n📍 Next step: python scripts/stage_3/step2_tokenize_dataset.py\n")

if __name__ == "__main__":
    train_tokenizer()
