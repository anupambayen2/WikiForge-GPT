# Save as verify_setup.py
import sys
print(f"Python version: {sys.version}")

# Check core packages
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers: {e}")

try:
    import tensorboard
    print(f"✓ TensorBoard installed")
except ImportError as e:
    print(f"✗ TensorBoard: {e}")

try:
    import lxml
    print(f"✓ lxml installed")
except ImportError as e:
    print(f"✗ lxml: {e}")

try:
    import wikiextractor
    print(f"✓ wikiextractor installed (bonus!)")
except ImportError as e:
    print(f"✗ wikiextractor: {e}")

print("\n✅ Setup verification complete!")