"""
WikiForge-GPT Extraction Validator
Checks random batch files to verify article format and quality
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import sys

class ExtractionValidator:
    def __init__(self, extracted_dir: str = r"E:\WikiForge-GPT\data\processed\extracted"):
        self.extracted_dir = Path(extracted_dir)
        self.issues = []
        
    def get_batch_files(self) -> List[Path]:
        """Get all batch files from extracted directory"""
        batch_files = list(self.extracted_dir.glob("articles_*.jsonl"))
        print(f"Found {len(batch_files)} batch files")
        return batch_files
    
    def validate_article(self, article: Dict[str, Any], file_name: str, line_num: int) -> bool:
        """Validate a single article structure"""
        required_fields = ['title', 'text']  # Simplified format from extraction
        optional_fields = ['id', 'url', 'categories', 'metadata']
        
        valid = True
        
        # Check required fields
        for field in required_fields:
            if field not in article:
                self.issues.append(f"❌ {file_name}:{line_num} - Missing required field: {field}")
                valid = False
        
        # Check field types and content
        if 'id' in article:
            if not isinstance(article['id'], (int, str)):
                self.issues.append(f"⚠️  {file_name}:{line_num} - 'id' should be int or str, got {type(article['id'])}")
        
        if 'title' in article:
            if not isinstance(article['title'], str):
                self.issues.append(f"⚠️  {file_name}:{line_num} - 'title' should be str, got {type(article['title'])}")
            elif len(article['title'].strip()) == 0:
                self.issues.append(f"⚠️  {file_name}:{line_num} - 'title' is empty")
        
        if 'text' in article:
            if not isinstance(article['text'], str):
                self.issues.append(f"⚠️  {file_name}:{line_num} - 'text' should be str, got {type(article['text'])}")
            elif len(article['text'].strip()) == 0:
                self.issues.append(f"⚠️  {file_name}:{line_num} - 'text' is empty")
        
        return valid
    
    def check_batch_file(self, file_path: Path, max_articles: int = 10) -> Dict[str, Any]:
        """Check a single batch file"""
        print(f"\n{'='*80}")
        print(f"Checking: {file_path.name}")
        print(f"{'='*80}")
        
        stats = {
            'file': file_path.name,
            'total_articles': 0,
            'valid_articles': 0,
            'invalid_articles': 0,
            'redirect_count': 0,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'sample_articles': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    stats['total_articles'] += 1
                    
                    try:
                        article = json.loads(line.strip())
                        
                        # Count redirects
                        text = article.get('text', '').strip()
                        if text.startswith('#REDIRECT') or text.startswith('#redirect'):
                            stats['redirect_count'] += 1
                        
                        # Validate article structure
                        if self.validate_article(article, file_path.name, line_num):
                            stats['valid_articles'] += 1
                        else:
                            stats['invalid_articles'] += 1
                        
                        # Store sample articles (first few)
                        if len(stats['sample_articles']) < max_articles:
                            is_redirect = article.get('text', '').strip().startswith('#REDIRECT') or article.get('text', '').strip().startswith('#redirect')
                            stats['sample_articles'].append({
                                'line': line_num,
                                'id': article.get('id', 'N/A'),
                                'title': article.get('title', 'N/A'),
                                'text_length': len(article.get('text', '')),
                                'text_preview': article.get('text', '')[:200] + '...' if len(article.get('text', '')) > 200 else article.get('text', ''),
                                'has_url': 'url' in article,
                                'has_categories': 'categories' in article,
                                'is_redirect': is_redirect,
                            })
                    
                    except json.JSONDecodeError as e:
                        self.issues.append(f"❌ {file_path.name}:{line_num} - JSON parse error: {e}")
                        stats['invalid_articles'] += 1
            
            # Print stats
            print(f"\n📊 File Statistics:")
            print(f"  • Total articles: {stats['total_articles']:,}")
            print(f"  • Valid articles: {stats['valid_articles']:,}")
            print(f"  • Invalid articles: {stats['invalid_articles']:,}")
            print(f"  • Redirect pages: {stats['redirect_count']:,} ({stats['redirect_count']/stats['total_articles']*100:.1f}%)")
            print(f"  • File size: {stats['file_size_mb']:.2f} MB")
            
            # Print sample articles
            print(f"\n📄 Sample Articles (first {len(stats['sample_articles'])}):")
            for sample in stats['sample_articles']:
                print(f"\n  Line {sample['line']}:")
                if sample['id'] != 'N/A':
                    print(f"    ID: {sample['id']}")
                print(f"    Title: {sample['title']}")
                print(f"    Text length: {sample['text_length']:,} chars")
                print(f"    Is redirect: {sample.get('is_redirect', False)}")
                print(f"    Preview: {sample['text_preview'][:150]}...")
        
        except Exception as e:
            self.issues.append(f"❌ Error reading {file_path.name}: {e}")
            print(f"❌ Error: {e}")
        
        return stats
    
    def run_validation(self, num_files: int = 5, articles_per_file: int = 10):
        """Run validation on random batch files"""
        print("\n" + "="*80)
        print("🔍 WikiForge-GPT Extraction Validator")
        print("="*80)
        
        # Get all batch files
        batch_files = self.get_batch_files()
        
        if not batch_files:
            print("❌ No batch files found!")
            return
        
        # Select random files
        num_to_check = min(num_files, len(batch_files))
        random_files = random.sample(batch_files, num_to_check)
        
        print(f"\n📋 Will check {num_to_check} random files out of {len(batch_files)} total")
        
        all_stats = []
        
        # Check each file
        for file_path in random_files:
            stats = self.check_batch_file(file_path, articles_per_file)
            all_stats.append(stats)
        
        # Summary report
        print("\n" + "="*80)
        print("📊 VALIDATION SUMMARY")
        print("="*80)
        
        total_articles = sum(s['total_articles'] for s in all_stats)
        total_valid = sum(s['valid_articles'] for s in all_stats)
        total_invalid = sum(s['invalid_articles'] for s in all_stats)
        total_redirects = sum(s['redirect_count'] for s in all_stats)
        total_size_mb = sum(s['file_size_mb'] for s in all_stats)
        
        print(f"\nFiles checked: {len(all_stats)}")
        print(f"Total articles checked: {total_articles:,}")
        print(f"Valid articles: {total_valid:,} ({total_valid/total_articles*100:.2f}%)")
        print(f"Invalid articles: {total_invalid:,} ({total_invalid/total_articles*100:.2f}%)")
        print(f"Redirect pages: {total_redirects:,} ({total_redirects/total_articles*100:.1f}%)")
        print(f"Total size checked: {total_size_mb:.2f} MB")
        
        # Issues report
        if self.issues:
            print(f"\n⚠️  Issues Found: {len(self.issues)}")
            print("\nFirst 20 issues:")
            for issue in self.issues[:20]:
                print(f"  {issue}")
            if len(self.issues) > 20:
                print(f"  ... and {len(self.issues) - 20} more issues")
        else:
            print("\n✅ No issues found! All checked articles are properly formatted.")
        
        # Recommendations
        print("\n" + "="*80)
        print("💡 RECOMMENDATIONS")
        print("="*80)
        
        if total_invalid == 0:
            print("✅ Extraction looks good! Safe to proceed with consolidation.")
        elif total_invalid / total_articles < 0.01:  # Less than 1% invalid
            print("⚠️  Small number of invalid articles detected.")
            print("   Consider reviewing the issues, but likely safe to proceed.")
        else:
            print("❌ Significant number of invalid articles!")
            print("   Review the extraction process before consolidating.")
        
        # Estimate full dataset
        avg_articles_per_file = total_articles / len(all_stats)
        estimated_total = avg_articles_per_file * len(batch_files)
        print(f"\n📈 Estimated total articles in all batches: {estimated_total:,.0f}")
        
        return all_stats

if __name__ == "__main__":
    # You can customize these parameters
    NUM_FILES_TO_CHECK = 10  # Number of random files to check
    ARTICLES_PER_FILE = 15   # Number of sample articles to display per file
    
    validator = ExtractionValidator()
    validator.run_validation(NUM_FILES_TO_CHECK, ARTICLES_PER_FILE)