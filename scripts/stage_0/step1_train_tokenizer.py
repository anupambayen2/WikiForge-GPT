"""
WikiForge-GPT Custom Tokenizer Trainer
=======================================
Train a BPE (Byte-Pair Encoding) tokenizer from scratch on Wikipedia data.

This creates YOUR OWN tokenizer vocabulary, not GPT-2's!
"""

import json
from pathlib import Path
from typing import List, Iterator
import sys

# Import centralized paths
PROJECT_ROOT = Path("E:/WikiForge-GPT")
sys.path.insert(0, str(PROJECT_ROOT))

from paths import PATHS
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tqdm import tqdm

class WikiTokenizerTrainer:
    """
    Train a custom BPE tokenizer on Wikipedia data.
    This creates a tokenizer optimized for YOUR specific dataset.
    """
    
    def __init__(
        self,
        vocab_size: int = 8000,  # Start small for Stage 0
        min_frequency: int = 2,
        special_tokens: List[str] = None,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens for our tokenizer
        if special_tokens is None:
            self.special_tokens = [
                "<|endoftext|>",  # End of document
                "<|pad|>",         # Padding
                "<|unk|>",         # Unknown token
            ]
        else:
            self.special_tokens = special_tokens
        
        # Paths from centralized config
        self.input_dir = PATHS.PROCESSED_DATA / "consolidated"
        self.output_dir = PATHS.TOKENIZER_DATA
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
    
    def article_iterator(self, max_articles: int = None) -> Iterator[str]:
        """
        Yield articles from train.jsonl for tokenizer training.
        
        Args:
            max_articles: Limit number of articles (None = all)
        """
        train_file = self.input_dir / "train.jsonl"
        
        print(f"\nReading articles from: {train_file}")
        
        count = 0
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading articles", unit=" articles"):
                if max_articles and count >= max_articles:
                    break
                
                article = json.loads(line.strip())
                yield article['text']
                
                count += 1
        
        print(f"Loaded {count:,} articles for tokenizer training")
    
    def train_tokenizer(self, training_articles: int = 100000):
        """
        Train the BPE tokenizer on Wikipedia articles.
        
        Args:
            training_articles: Number of articles to train on
                             (More = better vocabulary, but slower)
        """
        
        print("\n" + "="*80)
        print("🔧 Training Custom BPE Tokenizer")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Min frequency: {self.min_frequency}")
        print(f"  Training articles: {training_articles:,}")
        print(f"  Special tokens: {self.special_tokens}")
        
        # Initialize tokenizer with BPE model
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        
        # Use ByteLevel pre-tokenizer (like GPT-2, handles all Unicode)
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        
        # Configure BPE trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )
        
        # Collect articles for training
        print("\n📚 Collecting training data...")
        articles = list(self.article_iterator(max_articles=training_articles))
        
        print(f"\n🚀 Training tokenizer on {len(articles):,} articles...")
        print("This may take 10-30 minutes depending on data size...\n")
        
        # Train the tokenizer
        tokenizer.train_from_iterator(
            articles,
            trainer=trainer,
            length=len(articles),
        )
        
        # Add post-processor and decoder
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
        tokenizer.decoder = ByteLevelDecoder()
        
        # Save tokenizer
        output_path = self.output_dir / f"tokenizer_vocab{self.vocab_size}.json"
        tokenizer.save(str(output_path))
        
        print(f"\n✅ Tokenizer trained and saved to: {output_path}")
        
        # Test the tokenizer
        self.test_tokenizer(tokenizer)
        
        # Save metadata
        self.save_metadata(len(articles))
        
        return tokenizer
    
    def test_tokenizer(self, tokenizer: Tokenizer):
        """Test the trained tokenizer on sample text."""
        
        print("\n" + "="*80)
        print("🧪 Testing Tokenizer")
        print("="*80)
        
        test_texts = [
            "The history of artificial intelligence began in the 1950s.",
            "Wikipedia is a free online encyclopedia.",
            "Machine learning is a subset of artificial intelligence.",
        ]
        
        for text in test_texts:
            encoding = tokenizer.encode(text)
            decoded = tokenizer.decode(encoding.ids)
            
            print(f"\nOriginal: {text}")
            print(f"Tokens: {encoding.tokens}")
            print(f"Token IDs: {encoding.ids}")
            print(f"Decoded: {decoded}")
            print(f"Number of tokens: {len(encoding.ids)}")
    
    def save_metadata(self, num_articles: int):
        """Save tokenizer training metadata."""
        
        metadata = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'training_articles': num_articles,
            'tokenizer_type': 'BPE',
            'input_data': str(self.input_dir),
        }
        
        metadata_path = self.output_dir / f"tokenizer_vocab{self.vocab_size}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Metadata saved to: {metadata_path}")
    
    def load_tokenizer(self, vocab_size: int = None) -> Tokenizer:
        """Load a previously trained tokenizer."""
        
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        tokenizer_path = self.output_dir / f"tokenizer_vocab{vocab_size}.json"
        
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                f"Train it first using train_tokenizer()"
            )
        
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"✅ Loaded tokenizer from: {tokenizer_path}")
        
        return tokenizer


def main():
    """Main execution for tokenizer training."""
    
    print("\n" + "="*80)
    print("WikiForge-GPT Tokenizer Trainer")
    print("="*80)
    
    # Stage 0: Tiny model tokenizer (8K vocab)
    print("\n🎯 Training Stage 0 Tokenizer (8K vocabulary)")
    print("This will be used for the tiny model to learn the pipeline")
    
    trainer = WikiTokenizerTrainer(
        vocab_size=8000,
        min_frequency=2,
    )
    
    # Train on 100K articles (subset for speed)
    # For production, use 1M+ articles
    tokenizer = trainer.train_tokenizer(training_articles=100000)
    
    print("\n" + "="*80)
    print("✅ TOKENIZER TRAINING COMPLETE!")
    print("="*80)
    
    print(f"\n📁 Tokenizer saved in: {PATHS.TOKENIZER_DATA}")
    print(f"\n💡 Next Steps:")
    print(f"   1. Review tokenizer output in {PATHS.TOKENIZER_DATA}")
    print(f"   2. Test tokenizer with different vocab sizes if needed")
    print(f"   3. Proceed to tokenizing the full dataset")
    
    return tokenizer


if __name__ == "__main__":
    # Ensure paths exist
    PATHS.create_all()
    
    # Train tokenizer
    main()