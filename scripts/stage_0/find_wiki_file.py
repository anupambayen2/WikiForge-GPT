"""
Find the Wikipedia XML file in F:\wiki_raw
"""

from pathlib import Path

print("="*80)
print("SEARCHING FOR WIKIPEDIA FILE")
print("="*80)

wiki_dir = Path("F:/wiki_raw")

print(f"\nSearching in: {wiki_dir}")

if not wiki_dir.exists():
    print(f"✗ Directory does not exist: {wiki_dir}")
    print("\nPlease check:")
    print("1. Is the drive letter correct? (F:)")
    print("2. Is the folder name correct? (wiki_raw)")
else:
    print(f"✓ Directory exists: {wiki_dir}\n")
    
    # List all files
    print("Files in this directory:")
    print("-"*80)
    
    all_files = list(wiki_dir.glob("*"))
    
    if not all_files:
        print("✗ No files found in this directory!")
    else:
        for file in sorted(all_files):
            if file.is_file():
                size_gb = file.stat().st_size / 1e9
                print(f"  {file.name}")
                print(f"    Size: {size_gb:.2f} GB")
                print()
    
    # Look for XML files specifically
    xml_files = list(wiki_dir.glob("*.xml*"))
    
    if xml_files:
        print("\n" + "="*80)
        print("✓ FOUND WIKIPEDIA XML FILES:")
        print("="*80)
        for xml_file in xml_files:
            size_gb = xml_file.stat().st_size / 1e9
            print(f"\nFile: {xml_file.name}")
            print(f"Full path: {xml_file}")
            print(f"Size: {size_gb:.2f} GB")
    else:
        print("\n✗ No XML files found!")
        print("Please check if the Wikipedia dump is in this folder")

print("\n" + "="*80)