"""
Custom Wikipedia XML parser - handles .bz2 files directly.
"""

import bz2
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import json
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths import PATHS

def clean_text(text):
    """Remove Wikipedia markup and get clean text."""
    # Remove templates {{...}}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # Remove references <ref>...</ref>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove file/image links
    text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[Image:.*?\]\]', '', text, flags=re.IGNORECASE)
    # Convert wiki links [[link|text]] to just text
    text = re.sub(r'\[\[([^|\]]*\|)?([^\]]+)\]\]', r'\2', text)
    # Remove category links
    text = re.sub(r'\[\[Category:.*?\]\]', '', text, flags=re.IGNORECASE)
    # Remove bold/italic markup
    text = re.sub(r"'''?", '', text)
    # Clean up whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def extract_wikipedia():
    print("="*80)
    print("EXTRACTING WIKIPEDIA (CUSTOM PARSER)")
    print("="*80)
    
    input_file = Path("F:/wiki_raw/enwiki-latest-pages-articles.xml_2.bz2")
    output_dir = PATHS.PROCESSED_DATA / "extracted"
    
    if not input_file.exists():
        print(f"✗ File not found: {input_file}")
        sys.exit(1)
    
    print(f"\n✓ Input: {input_file}")
    print(f"✓ Size: {input_file.stat().st_size / 1e9:.2f} GB (compressed)")
    print(f"✓ Output: {output_dir}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    articles = []
    article_count = 0
    file_count = 0
    
    print("Opening compressed file (this may take a moment)...")
    
    with bz2.open(input_file, 'rt', encoding='utf-8', errors='ignore') as f:
        current_title = None
        current_text = []
        in_text = False
        
        print("Processing articles (this will take 30-60 minutes)...")
        print("Progress updates every 1000 articles:")
        print("-"*80)
        
        line_count = 0
        
        for line in f:
            line_count += 1
            line = line.strip()
            
            # Extract title
            if '<title>' in line:
                match = re.search(r'<title>(.*?)</title>', line)
                if match:
                    current_title = match.group(1)
            
            # Track text section
            if '<text' in line:
                in_text = True
                # Extract text from same line
                match = re.search(r'<text[^>]*>(.*)', line)
                if match:
                    current_text.append(match.group(1))
                continue
            
            if '</text>' in line:
                in_text = False
                # Get remaining text
                match = re.search(r'(.*)</text>', line)
                if match:
                    current_text.append(match.group(1))
                
                # Process article
                if current_title and current_text:
                    text = ' '.join(current_text)
                    cleaned = clean_text(text)
                    
                    # Filter out unwanted pages
                    skip_prefixes = ('Wikipedia:', 'Talk:', 'User:', 'Template:', 
                                   'Help:', 'File:', 'MediaWiki:', 'Category:')
                    
                    if (len(cleaned) >= 100 and 
                        not current_title.startswith(skip_prefixes) and
                        '(disambiguation)' not in current_title.lower()):
                        
                        articles.append({
                            'title': current_title,
                            'text': cleaned
                        })
                        article_count += 1
                        
                        # Progress update
                        if article_count % 1000 == 0:
                            print(f"  Processed {article_count:,} articles...")
                        
                        # Save in batches of 10,000 articles
                        if len(articles) >= 10000:
                            output_file = output_dir / f"articles_{file_count:04d}.jsonl"
                            with open(output_file, 'w', encoding='utf-8') as out:
                                for article in articles:
                                    json.dump(article, out, ensure_ascii=False)
                                    out.write('\n')
                            
                            print(f"  → Saved batch {file_count} ({len(articles)} articles)")
                            articles = []
                            file_count += 1
                
                # Reset for next article
                current_title = None
                current_text = []
                continue
            
            # Collect text
            if in_text:
                current_text.append(line)
        
        # Save remaining articles
        if articles:
            output_file = output_dir / f"articles_{file_count:04d}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as out:
                for article in articles:
                    json.dump(article, out, ensure_ascii=False)
                    out.write('\n')
            print(f"  → Saved final batch {file_count} ({len(articles)} articles)")
    
    print("\n" + "="*80)
    print("✓ EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\n✓ Total articles extracted: {article_count:,}")
    print(f"✓ Output files created: {file_count + 1}")
    print(f"✓ Output location: {output_dir}")
    
    # Estimate tokens
    print(f"\nEstimated tokens: ~{article_count * 500:,} (rough estimate)")
    
    print("\n" + "="*80)
    print("NEXT STEP:")
    print("="*80)
    print("Run: python scripts\\consolidate_articles.py")
    print("="*80)

if __name__ == "__main__":
    extract_wikipedia()