"""
STAGE_3 - Step 2: Tokenize Dataset (MEMORY-EFFICIENT - FIXED)
===============================================================================
Tokenize dataset using CHUNKING to avoid memory errors.
Uses PROVEN approach from Stage 1!
"""

import json
from pathlib import Path
from typing import Dict, List
import sys
import numpy as np
from tqdm import tqdm
import gc

PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))

from paths import PATHS
from tokenizers import Tokenizer

class MemoryEfficientTokenizer:
    """
    Tokenize dataset in chunks to avoid memory issues.
    Uses same proven approach as Stage 1!
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_seq_length: int = 256,
        stride: int = 128,
        chunk_size: int = 10000,  # Process 10K articles at a time
    ):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.stride = stride
        self.chunk_size = chunk_size
        
        # Paths
        self.input_dir = PATHS.PROCESSED_DATA / "consolidated"
        self.tokenizer_path = PATHS.TOKENIZER_DATA / f"tokenizer_vocab{vocab_size}.json"
        self.output_dir = PATHS.TRAINING_DATA / f"stage_3_vocab{vocab_size}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer
        print(f"📂 Loading tokenizer: {self.tokenizer_path.name}")
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        
        # Get special token IDs
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        
        print(f"✅ Tokenizer loaded!")
        print(f"   Vocab size: {self.tokenizer.get_vocab_size():,}")
        print(f"   PAD token ID: {self.pad_token_id}")
        print(f"   EOS token ID: {self.eos_token_id}")
        print(f"   Chunk size: {self.chunk_size:,} articles")
    
    def tokenize_article(self, text: str) -> List[List[int]]:
        """
        Tokenize a single article and split into sequences.
        Returns list of token sequences.
        """
        # Tokenize the text
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        
        # Add EOS token at the end
        token_ids.append(self.eos_token_id)
        
        # If article fits in one sequence, return it
        if len(token_ids) <= self.max_seq_length:
            # Pad to max length
            padded = token_ids + [self.pad_token_id] * (self.max_seq_length - len(token_ids))
            return [padded]
        
        # Otherwise, create overlapping windows
        sequences = []
        start = 0
        
        while start < len(token_ids):
            end = min(start + self.max_seq_length, len(token_ids))
            sequence = token_ids[start:end]
            
            # Pad if needed (last sequence might be shorter)
            if len(sequence) < self.max_seq_length:
                sequence = sequence + [self.pad_token_id] * (self.max_seq_length - len(sequence))
            
            sequences.append(sequence)
            
            # Move window with stride
            start += self.stride
            
            # Stop if we've covered the whole article
            if end == len(token_ids):
                break
        
        return sequences
    
    def process_chunk(self, articles: List[Dict]) -> np.ndarray:
        """Process a chunk of articles and return numpy array."""
        chunk_sequences = []
        
        for article in articles:
            text = article['text']
            sequences = self.tokenize_article(text)
            chunk_sequences.extend(sequences)
        
        # Convert to numpy array
        return np.array(chunk_sequences, dtype=np.int32)
    
    def process_split(self, split_name: str) -> Dict:
        """
        Process a single data split in chunks.
        Saves to disk incrementally.
        """
        
        print("\n" + "="*80)
        print(f"Processing {split_name.upper()} Split")
        print("="*80)
        
        input_file = self.input_dir / f"{split_name}.jsonl"
        output_file = self.output_dir / f"{split_name}_sequences.npy"
        
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        
        # Count total articles first
        print("\nCounting articles...")
        with open(input_file, 'r', encoding='utf-8') as f:
            total_articles = sum(1 for _ in f)
        
        print(f"Total articles: {total_articles:,}")
        print(f"Processing in chunks of {self.chunk_size:,}")
        
        # Statistics
        article_count = 0
        sequence_count = 0
        chunk_files = []
        
        # Process in chunks
        print(f"\nTokenizing and saving chunks...")
        
        chunk_articles = []
        chunk_num = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=total_articles, desc=f"Processing {split_name}")
            
            for line in f:
                article = json.loads(line.strip())
                chunk_articles.append(article)
                article_count += 1
                
                # Process when chunk is full
                if len(chunk_articles) >= self.chunk_size:
                    # Tokenize chunk
                    chunk_array = self.process_chunk(chunk_articles)
                    sequence_count += len(chunk_array)
                    
                    # Save chunk to disk
                    chunk_file = self.output_dir / f"{split_name}_chunk_{chunk_num:04d}.npy"
                    np.save(chunk_file, chunk_array)
                    chunk_files.append(chunk_file)
                    
                    # Update progress
                    pbar.update(len(chunk_articles))
                    pbar.set_postfix({
                        'chunk': chunk_num,
                        'sequences': f"{sequence_count:,}",
                        'memory': f"{chunk_array.nbytes / 1024 / 1024:.1f}MB"
                    })
                    
                    # Clear memory
                    del chunk_array
                    chunk_articles = []
                    chunk_num += 1
                    gc.collect()
            
            # Process remaining articles
            if chunk_articles:
                chunk_array = self.process_chunk(chunk_articles)
                sequence_count += len(chunk_array)
                
                chunk_file = self.output_dir / f"{split_name}_chunk_{chunk_num:04d}.npy"
                np.save(chunk_file, chunk_array)
                chunk_files.append(chunk_file)
                
                pbar.update(len(chunk_articles))
                
                del chunk_array
                gc.collect()
            
            pbar.close()
        
        # Concatenate all chunks into final file
        print(f"\nCombining {len(chunk_files)} chunks into final file...")
        
        # Use memory-mapped array for efficient combination
        # First, get total size
        total_sequences = 0
        for chunk_file in chunk_files:
            chunk = np.load(chunk_file, mmap_mode='r')
            total_sequences += len(chunk)
        
        print(f"Total sequences: {total_sequences:,}")
        print(f"Creating memory-mapped output file...")
        
        # Create memory-mapped output file
        final_shape = (total_sequences, self.max_seq_length)
        mmap_array = np.memmap(
            output_file,
            dtype=np.int32,
            mode='w+',
            shape=final_shape
        )
        
        # Copy chunks into final file
        current_idx = 0
        for chunk_file in tqdm(chunk_files, desc="Combining chunks"):
            chunk = np.load(chunk_file)
            chunk_size = len(chunk)
            mmap_array[current_idx:current_idx + chunk_size] = chunk
            current_idx += chunk_size
            
            # Delete chunk file
            chunk_file.unlink()
        
        # Flush to disk
        del mmap_array
        
        print(f"✅ Final file saved: {output_file}")
        
        # Calculate file size
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        # Statistics
        stats = {
            'split': split_name,
            'articles': article_count,
            'sequences': sequence_count,
            'total_tokens': sequence_count * self.max_seq_length,
            'avg_sequences_per_article': sequence_count / article_count,
            'file_size_mb': file_size_mb,
        }
        
        # Save stats
        stats_file = self.output_dir / f"{split_name}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print(f"\n📊 {split_name.upper()} Statistics:")
        print(f"   Articles processed: {stats['articles']:,}")
        print(f"   Sequences created: {stats['sequences']:,}")
        print(f"   Total tokens: {stats['total_tokens']:,}")
        print(f"   Avg sequences/article: {stats['avg_sequences_per_article']:.2f}")
        print(f"   File size: {stats['file_size_mb']:.2f} MB")
        
        return stats
    
    def run(self):
        """Process all data splits."""
        
        print("\n" + "="*80)
        print("🚀 STAGE 3 Dataset Tokenization (Memory-Efficient)")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Max sequence length: {self.max_seq_length}")
        print(f"  Stride (overlap): {self.stride}")
        print(f"  Chunk size: {self.chunk_size:,} articles")
        print(f"  Output directory: {self.output_dir}")
        
        # Process each split
        all_stats = {}
        
        for split in ['train', 'val', 'test']:
            stats = self.process_split(split)
            all_stats[split] = stats
        
        # Final summary
        print("\n" + "="*80)
        print("✅ DATASET TOKENIZATION COMPLETE!")
        print("="*80)
        
        total_sequences = sum(s['sequences'] for s in all_stats.values())
        total_tokens = sum(s['total_tokens'] for s in all_stats.values())
        total_size_mb = sum(s['file_size_mb'] for s in all_stats.values())
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total sequences: {total_sequences:,}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
        
        print(f"\n📁 Tokenized data saved to:")
        print(f"   {self.output_dir}")
        
        # Save overall stats
        overall_stats = {
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'stride': self.stride,
            'chunk_size': self.chunk_size,
            'splits': all_stats,
            'total_sequences': total_sequences,
            'total_tokens': total_tokens,
        }
        
        overall_stats_file = self.output_dir / "dataset_stats.json"
        with open(overall_stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        print(f"\n📍 Next step: python scripts/stage_3/step3_train_model.py")
        
        return all_stats


def main():
    """Main execution."""
    
    print("\n🎯 Tokenizing for Stage 3 (350M Model)")
    print("   Vocab: 50K | Sequence Length: 256 tokens")
    print("   Processing in chunks to avoid memory issues")
    
    tokenizer = MemoryEfficientTokenizer(
        vocab_size=50000,
        max_seq_length=256,
        stride=128,
        chunk_size=10000,  # 10K articles per chunk
    )
    
    stats = tokenizer.run()
    
    return stats


if __name__ == "__main__":
    PATHS.create_all()
    main()