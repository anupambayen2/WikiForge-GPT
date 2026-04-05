"""
STAGE_2 - Step 2: Tokenize Dataset
===============================================================================
Tokenize 2.88M Wikipedia articles with 32,000 token vocabulary
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
    print("\n" + "="*80)
    print("STAGE_2 - TOKENIZE DATASET")
    print("="*80 + "\n")
    
    # Config
    max_seq_length = 256
    stride = 128
    chunk_size = 10000
    vocab_size = 32000
    
    # Load tokenizer
    tokenizer_path = PATHS.TOKENIZER_DATA / f"tokenizer_vocab{vocab_size}.json"
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print("   Please run step1_train_tokenizer.py first!")
        return
    
    print(f"📂 Loading tokenizer: {tokenizer_path.name}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    
    print(f"✅ Tokenizer loaded")
    print(f"   Vocab size: {tokenizer.get_vocab_size():,}")
    print()
    
    # Output directory
    output_dir = PATHS.TRAINING_DATA / "stage_2_vocab32000"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⚙️  Configuration:")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Stride: {stride}")
    print(f"   Output: {output_dir}")
    print()
    
    # Process splits
    for split in ['train', 'val', 'test']:
        input_file = PATHS.PROCESSED_DATA / "consolidated" / f"{split}.jsonl"
        if not input_file.exists():
            print(f"⚠️  Skipping {split}: file not found")
            continue
        
        print(f"Processing {split.upper()}...")
        all_sequences = []
        
        total_articles = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        
        with open(input_file, 'r', encoding='utf-8') as f:
            chunk = []
            for line in tqdm(f, total=total_articles, desc=f"Tokenizing {split}"):
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
        output_file = output_dir / f"{split}_sequences.npy"
        np.save(output_file, sequences_array)
        
        file_size_gb = output_file.stat().st_size / 1e9
        print(f"✅ {split.upper()}: {len(sequences_array):,} sequences, {file_size_gb:.2f} GB")
    
    print(f"\n✅ COMPLETE!")
    print(f"Data saved to: {output_dir}")
    print(f"\n📍 Next step: python scripts/stage_2/step3_train_model.py\n")

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
