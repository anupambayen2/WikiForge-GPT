"""
Diagnostic script to find extracted files
"""

from pathlib import Path
import os

extracted_dir = Path(r"E:\WikiForge-GPT\data\processed\extracted")

print("="*80)
print("🔍 Checking extracted directory...")
print("="*80)

print(f"\nLooking in: {extracted_dir}")
print(f"Directory exists: {extracted_dir.exists()}")

if extracted_dir.exists():
    print(f"Is directory: {extracted_dir.is_dir()}")
    
    # List all files
    all_files = list(extracted_dir.iterdir())
    print(f"\nTotal items in directory: {len(all_files)}")
    
    # Categorize by type
    files = [f for f in all_files if f.is_file()]
    dirs = [f for f in all_files if f.is_dir()]
    
    print(f"Files: {len(files)}")
    print(f"Directories: {len(dirs)}")
    
    if files:
        print(f"\n📄 First 20 files:")
        for i, f in enumerate(files[:20], 1):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {i}. {f.name} ({size_mb:.2f} MB)")
        
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more files")
        
        # Check file extensions
        extensions = {}
        for f in files:
            ext = f.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        print(f"\n📊 File extensions:")
        for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ext or '(no extension)'}: {count} files")
        
        # Look for common patterns
        print(f"\n🔍 Looking for common patterns:")
        patterns = [
            ("articles_batch_*.jsonl", "articles_batch_"),
            ("batch_*.jsonl", "batch_"),
            ("*.jsonl", ".jsonl"),
            ("*.json", ".json"),
            ("articles_*.jsonl", "articles_"),
        ]
        
        for pattern_desc, pattern in patterns:
            matching = [f for f in files if pattern in f.name.lower()]
            if matching:
                print(f"  ✅ Found {len(matching)} files matching pattern: {pattern_desc}")
                print(f"     Example: {matching[0].name}")
            else:
                print(f"  ❌ No files matching: {pattern_desc}")
    else:
        print("\n❌ No files found in directory!")
    
    if dirs:
        print(f"\n📁 Subdirectories:")
        for d in dirs[:10]:
            print(f"  - {d.name}")

else:
    print("\n❌ Directory does not exist!")
    print("\nPossible locations to check:")
    base = Path(r"E:\WikiForge-GPT")
    for subdir in ['data', 'processed', 'output', 'outputs']:
        check_path = base / subdir
        if check_path.exists():
            print(f"  ✅ Found: {check_path}")
            # List contents
            contents = list(check_path.iterdir())
            if contents:
                print(f"     Contains: {', '.join([c.name for c in contents[:5]])}")
        else:
            print(f"  ❌ Not found: {check_path}")