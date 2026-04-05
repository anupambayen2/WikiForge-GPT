"""
Inspect raw JSON structure of extracted articles
"""

import json
from pathlib import Path

extracted_dir = Path(r"E:\WikiForge-GPT\data\processed\extracted")

print("="*80)
print("🔍 Inspecting Raw JSON Structure")
print("="*80)

# Get first file
files = sorted(extracted_dir.glob("articles_*.jsonl"))
if files:
    first_file = files[0]
    print(f"\nReading: {first_file.name}")
    
    with open(first_file, 'r', encoding='utf-8') as f:
        # Read first 5 lines
        for i in range(5):
            line = f.readline().strip()
            if line:
                print(f"\n{'='*80}")
                print(f"Article {i+1}:")
                print(f"{'='*80}")
                
                # Parse JSON
                try:
                    article = json.loads(line)
                    
                    # Show raw structure
                    print(f"\nRaw JSON keys: {list(article.keys())}")
                    
                    # Show each field
                    for key, value in article.items():
                        if isinstance(value, str):
                            preview = value[:100] + "..." if len(value) > 100 else value
                            print(f"\n{key}:")
                            print(f"  Type: {type(value).__name__}")
                            print(f"  Length: {len(value)} chars")
                            print(f"  Preview: {preview}")
                        else:
                            print(f"\n{key}: {value}")
                    
                    # Show formatted JSON
                    print(f"\nFormatted JSON (first 500 chars):")
                    formatted = json.dumps(article, indent=2, ensure_ascii=False)
                    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
                    
                except json.JSONDecodeError as e:
                    print(f"ERROR parsing JSON: {e}")
                    print(f"Raw line (first 200 chars): {line[:200]}")
else:
    print("No files found!")