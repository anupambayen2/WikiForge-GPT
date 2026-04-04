"""
WikiForge-GPT Smart Consolidation Script
Filters, cleans, and processes Wikipedia articles for LLM training
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime
import html

class SmartConsolidator:
    def __init__(
        self,
        input_dir: str = r"E:\WikiForge-GPT\data\processed\extracted",
        output_dir: str = r"E:\WikiForge-GPT\data\processed\consolidated"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_articles': 0,
            'redirects_filtered': 0,
            'short_articles_filtered': 0,
            'malformed_filtered': 0,
            'cleaned_articles': 0,
            'total_chars': 0,
            'total_words': 0,
        }
        
        # Configuration
        self.min_text_length = 100  # Minimum characters after cleaning
        self.min_words = 20  # Minimum words
        
    def is_redirect(self, text: str) -> bool:
        """Check if article is a redirect"""
        text_lower = text.strip().lower()
        return text_lower.startswith('#redirect') or text_lower.startswith('#re direct')
    
    def clean_wikitext(self, text: str) -> str:
        """Clean Wikipedia markup from text"""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove the embedded hash/metadata strings that appear in redirects
        # Pattern: long alphanumeric strings like "k93da0q0m9w7g593v22ktntmmw4redl"
        text = re.sub(r'\b[a-z0-9]{27,35}\b', '', text)
        
        # Remove Wikipedia-specific patterns
        # Remove file/image references
        text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\[Image:.*?\]\]', '', text, flags=re.IGNORECASE)
        
        # Remove category links
        text = re.sub(r'\[\[Category:.*?\]\]', '', text, flags=re.IGNORECASE)
        
        # Remove references
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<ref[^>]*/?>', '', text, flags=re.IGNORECASE)
        
        # Remove comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove template transclusions (basic - this is complex in wiki syntax)
        # Remove simple {{template}} patterns
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        
        # Convert wiki links [[link|text]] to just text, [[link]] to link
        text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)  # [[link|text]] -> text
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[link]] -> link
        
        # Remove external links markup but keep the text
        text = re.sub(r'\[https?://[^\s\]]+ ([^\]]+)\]', r'\1', text)
        text = re.sub(r'\[https?://[^\s\]]+\]', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove wiki formatting
        text = re.sub(r"'{2,}", '', text)  # Remove bold/italic markers
        
        # Remove table markup
        text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
        text = re.sub(r'^\s*\|.*?$', '', text, flags=re.MULTILINE)
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        return text
    
    def is_valid_article(self, title: str, text: str) -> tuple[bool, Optional[str]]:
        """
        Check if article meets quality criteria
        Returns (is_valid, reason_if_invalid)
        """
        
        # Check for redirect
        if self.is_redirect(text):
            return False, 'redirect'
        
        # Check minimum length
        if len(text) < self.min_text_length:
            return False, 'too_short'
        
        # Count words
        words = text.split()
        if len(words) < self.min_words:
            return False, 'too_few_words'
        
        # Check for malformed content
        # If text is mostly just metadata/markup after cleaning
        if len(text.strip()) == 0:
            return False, 'malformed'
        
        # Check if title is reasonable
        if not title or len(title) == 0:
            return False, 'no_title'
        
        return True, None
    
    def process_article(self, article: Dict[str, Any], article_id: int) -> Optional[Dict[str, Any]]:
        """Process a single article"""
        
        title = article.get('title', '')
        text = article.get('text', '')
        
        self.stats['total_articles'] += 1
        
        # Check validity
        is_valid, reason = self.is_valid_article(title, text)
        
        if not is_valid:
            if reason == 'redirect':
                self.stats['redirects_filtered'] += 1
            elif reason in ['too_short', 'too_few_words']:
                self.stats['short_articles_filtered'] += 1
            else:
                self.stats['malformed_filtered'] += 1
            return None
        
        # Clean the text
        cleaned_text = self.clean_wikitext(text)
        
        # Re-check length after cleaning
        if len(cleaned_text) < self.min_text_length:
            self.stats['short_articles_filtered'] += 1
            return None
        
        # Count words and chars
        word_count = len(cleaned_text.split())
        char_count = len(cleaned_text)
        
        if word_count < self.min_words:
            self.stats['short_articles_filtered'] += 1
            return None
        
        # Create processed article
        processed = {
            'id': article_id,
            'title': title,
            'text': cleaned_text,
            'metadata': {
                'word_count': word_count,
                'char_count': char_count,
                'original_length': len(text)
            }
        }
        
        self.stats['cleaned_articles'] += 1
        self.stats['total_chars'] += char_count
        self.stats['total_words'] += word_count
        
        return processed
    
    def process_batch_file(self, file_path: Path, start_id: int) -> tuple[list, int]:
        """Process a single batch file"""
        articles = []
        current_id = start_id
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    processed = self.process_article(article, current_id)
                    
                    if processed:
                        articles.append(processed)
                        current_id += 1
                
                except json.JSONDecodeError:
                    self.stats['malformed_filtered'] += 1
                    continue
        
        return articles, current_id
    
    def save_dataset(self, articles: list, split: str):
        """Save articles to file"""
        output_file = self.output_dir / f"{split}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        print(f"  ✅ Saved {len(articles):,} articles to {split}.jsonl")
    
    def create_splits(self, all_articles: list):
        """Create train/val/test splits"""
        
        total = len(all_articles)
        
        # Standard splits: 98% train, 1% val, 1% test
        val_size = int(total * 0.01)
        test_size = int(total * 0.01)
        train_size = total - val_size - test_size
        
        # Split the data
        train = all_articles[:train_size]
        val = all_articles[train_size:train_size + val_size]
        test = all_articles[train_size + val_size:]
        
        print(f"\n📊 Creating Dataset Splits:")
        print(f"  Train: {len(train):,} articles ({len(train)/total*100:.1f}%)")
        print(f"  Val:   {len(val):,} articles ({len(val)/total*100:.1f}%)")
        print(f"  Test:  {len(test):,} articles ({len(test)/total*100:.1f}%)")
        
        # Save splits
        self.save_dataset(train, 'train')
        self.save_dataset(val, 'val')
        self.save_dataset(test, 'test')
        
        # Also save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_articles': total,
            'splits': {
                'train': len(train),
                'val': len(val),
                'test': len(test)
            },
            'statistics': self.stats,
            'config': {
                'min_text_length': self.min_text_length,
                'min_words': self.min_words,
            }
        }
        
        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def run(self, batch_size: int = 100000):
        """
        Run the consolidation process
        batch_size: Save intermediate results every N articles
        """
        
        print("="*80)
        print("🚀 WikiForge-GPT Smart Consolidation")
        print("="*80)
        
        # Get all batch files
        batch_files = sorted(self.input_dir.glob("articles_*.jsonl"))
        print(f"\n📁 Found {len(batch_files)} batch files")
        
        # Process files
        all_articles = []
        current_id = 0
        start_time = time.time()
        
        print(f"\n⚙️  Processing articles...")
        print(f"  Filters: redirects, min {self.min_text_length} chars, min {self.min_words} words")
        print(f"  Cleaning: markup, templates, references, HTML entities\n")
        
        for i, file_path in enumerate(batch_files, 1):
            file_start = time.time()
            
            # Process batch file
            articles, current_id = self.process_batch_file(file_path, current_id)
            all_articles.extend(articles)
            
            # Progress update
            elapsed = time.time() - start_time
            articles_per_sec = self.stats['total_articles'] / elapsed if elapsed > 0 else 0
            
            print(f"  [{i}/{len(batch_files)}] {file_path.name}")
            print(f"    Processed: {self.stats['total_articles']:,} | "
                  f"Kept: {self.stats['cleaned_articles']:,} | "
                  f"Filtered: {self.stats['total_articles'] - self.stats['cleaned_articles']:,} | "
                  f"Speed: {articles_per_sec:.0f} articles/sec")
            
            # Save intermediate results every batch_size articles to manage memory
            if len(all_articles) >= batch_size:
                print(f"\n  💾 Saving intermediate batch ({len(all_articles):,} articles)...")
                temp_file = self.output_dir / f"temp_batch_{i}.jsonl"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    for article in all_articles:
                        f.write(json.dumps(article, ensure_ascii=False) + '\n')
                all_articles = []  # Clear memory
        
        # Collect all articles from temp files
        print(f"\n📦 Consolidating all batches...")
        final_articles = []
        
        # Add remaining articles
        final_articles.extend(all_articles)
        
        # Add articles from temp files
        temp_files = sorted(self.output_dir.glob("temp_batch_*.jsonl"))
        for temp_file in temp_files:
            with open(temp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    final_articles.append(json.loads(line.strip()))
            temp_file.unlink()  # Delete temp file
        
        # Print final statistics
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("📊 CONSOLIDATION COMPLETE")
        print("="*80)
        
        print(f"\n⏱️  Processing Time: {total_time/60:.1f} minutes")
        print(f"  Speed: {self.stats['total_articles']/total_time:.0f} articles/second")
        
        print(f"\n📈 Article Statistics:")
        print(f"  Total processed: {self.stats['total_articles']:,}")
        print(f"  ✅ Kept: {self.stats['cleaned_articles']:,} "
              f"({self.stats['cleaned_articles']/self.stats['total_articles']*100:.1f}%)")
        print(f"  ❌ Filtered: {self.stats['total_articles'] - self.stats['cleaned_articles']:,} "
              f"({(self.stats['total_articles'] - self.stats['cleaned_articles'])/self.stats['total_articles']*100:.1f}%)")
        
        print(f"\n🗑️  Filtering Breakdown:")
        print(f"  Redirects: {self.stats['redirects_filtered']:,}")
        print(f"  Too short: {self.stats['short_articles_filtered']:,}")
        print(f"  Malformed: {self.stats['malformed_filtered']:,}")
        
        print(f"\n📝 Content Statistics:")
        avg_words = self.stats['total_words'] / self.stats['cleaned_articles'] if self.stats['cleaned_articles'] > 0 else 0
        avg_chars = self.stats['total_chars'] / self.stats['cleaned_articles'] if self.stats['cleaned_articles'] > 0 else 0
        print(f"  Total words: {self.stats['total_words']:,}")
        print(f"  Total characters: {self.stats['total_chars']:,}")
        print(f"  Average words/article: {avg_words:.0f}")
        print(f"  Average chars/article: {avg_chars:.0f}")
        
        # Estimate tokens (rough: ~4 chars per token)
        estimated_tokens = self.stats['total_chars'] / 4
        print(f"  Estimated tokens: {estimated_tokens:,.0f}")
        
        # Create train/val/test splits
        self.create_splits(final_articles)
        
        print(f"\n✅ Output location: {self.output_dir}")
        print("="*80)
        
        return final_articles

if __name__ == "__main__":
    consolidator = SmartConsolidator()
    consolidator.run(batch_size=100000)