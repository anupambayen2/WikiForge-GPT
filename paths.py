"""
WikiForge-GPT Path Management
==============================
Centralized path definitions to be imported by all modules.
This solves the path problem by providing a single source of truth.

Usage:
    from paths import PATHS
    
    # Access any path
    data_file = PATHS.PROCESSED_DATA / "train.txt"
    model_checkpoint = PATHS.CHECKPOINTS / "stage_1" / "checkpoint_1000.pt"
"""

from pathlib import Path
import sys

# ============================================================================
# ROOT DIRECTORIES (ABSOLUTE PATHS)
# ============================================================================

PROJECT_ROOT = Path("E:/WikiForge-GPT")
DATA_ROOT = Path("F:/wiki_raw")

# ============================================================================
# ENSURE PROJECT ROOT IS IN PYTHON PATH
# ============================================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# PATHS CLASS (CENTRALIZED PATH MANAGEMENT)
# ============================================================================

class Paths:
    """
    Centralized path management for WikiForge-GPT project.
    All paths are defined here to avoid path-related issues.
    """
    
    # ========== ROOT DIRECTORIES ==========
    PROJECT_ROOT = PROJECT_ROOT
    DATA_ROOT = DATA_ROOT
    
    # ========== DATA DIRECTORIES ==========
    # Raw data
    RAW_DATA = DATA_ROOT / "raw"
    RAW_WIKI_FILE = DATA_ROOT / "enwiki-latest-pages-articles.xml_2"
    
    # Processed data
    DATA = PROJECT_ROOT / "data"
    PROCESSED_DATA = DATA / "processed"
    TOKENIZER_DATA = DATA / "tokenizer"
    TRAINING_DATA = DATA / "training"
    VALIDATION_DATA = DATA / "validation"
    TEST_DATA = DATA / "test"
    
    # Data subfolders by stage
    STAGE_DATA = DATA / "stages"
    
    # ========== MODEL DIRECTORIES ==========
    MODELS = PROJECT_ROOT / "models"
    CHECKPOINTS = MODELS / "checkpoints"
    BEST_MODELS = MODELS / "best"
    PRETRAINED = MODELS / "pretrained"
    EXPORTS = MODELS / "exports"
    
    # ========== SOURCE CODE ==========
    SRC = PROJECT_ROOT / "src"
    SRC_DATA = SRC / "data"
    SRC_MODEL = SRC / "model"
    SRC_TRAINING = SRC / "training"
    SRC_INFERENCE = SRC / "inference"
    SRC_UTILS = SRC / "utils"
    
    # ========== LOGS & MONITORING ==========
    LOGS = PROJECT_ROOT / "logs"
    TENSORBOARD = LOGS / "tensorboard"
    METRICS = LOGS / "metrics"
    TRAINING_LOGS = LOGS / "training"
    ERROR_LOGS = LOGS / "errors"
    
    # ========== OUTPUTS ==========
    OUTPUTS = PROJECT_ROOT / "outputs"
    PREDICTIONS = OUTPUTS / "predictions"
    SAMPLES = OUTPUTS / "samples"
    EVALUATIONS = OUTPUTS / "evaluations"
    VISUALIZATIONS = OUTPUTS / "visualizations"
    
    # ========== CONFIGURATIONS ==========
    CONFIGS = PROJECT_ROOT / "configs"
    STAGE_CONFIGS = CONFIGS / "stages"
    MODEL_CONFIGS = CONFIGS / "models"
    
    # ========== SCRIPTS & NOTEBOOKS ==========
    SCRIPTS = PROJECT_ROOT / "scripts"
    NOTEBOOKS = PROJECT_ROOT / "notebooks"
    TESTS = PROJECT_ROOT / "tests"
    
    # ========== CACHE & TEMP ==========
    CACHE = PROJECT_ROOT / ".cache"
    TEMP = PROJECT_ROOT / "temp"
    
    # ========== DOCUMENTATION ==========
    DOCS = PROJECT_ROOT / "docs"
    
    @classmethod
    def get_stage_checkpoint_dir(cls, stage_name: str) -> Path:
        """Get checkpoint directory for a specific stage."""
        return cls.CHECKPOINTS / stage_name
    
    @classmethod
    def get_stage_data_dir(cls, stage_name: str) -> Path:
        """Get data directory for a specific stage."""
        return cls.STAGE_DATA / stage_name
    
    @classmethod
    def get_stage_log_dir(cls, stage_name: str) -> Path:
        """Get log directory for a specific stage."""
        return cls.TRAINING_LOGS / stage_name
    
    @classmethod
    def get_stage_tensorboard_dir(cls, stage_name: str) -> Path:
        """Get tensorboard directory for a specific stage."""
        return cls.TENSORBOARD / stage_name
    
    @classmethod
    def create_all(cls):
        """Create all project directories."""
        directories = [
            # Data
            cls.RAW_DATA,
            cls.PROCESSED_DATA,
            cls.TOKENIZER_DATA,
            cls.TRAINING_DATA,
            cls.VALIDATION_DATA,
            cls.TEST_DATA,
            cls.STAGE_DATA,
            
            # Models
            cls.MODELS,
            cls.CHECKPOINTS,
            cls.BEST_MODELS,
            cls.PRETRAINED,
            cls.EXPORTS,
            
            # Source
            cls.SRC,
            cls.SRC_DATA,
            cls.SRC_MODEL,
            cls.SRC_TRAINING,
            cls.SRC_INFERENCE,
            cls.SRC_UTILS,
            
            # Logs
            cls.LOGS,
            cls.TENSORBOARD,
            cls.METRICS,
            cls.TRAINING_LOGS,
            cls.ERROR_LOGS,
            
            # Outputs
            cls.OUTPUTS,
            cls.PREDICTIONS,
            cls.SAMPLES,
            cls.EVALUATIONS,
            cls.VISUALIZATIONS,
            
            # Configs
            cls.CONFIGS,
            cls.STAGE_CONFIGS,
            cls.MODEL_CONFIGS,
            
            # Scripts
            cls.SCRIPTS,
            cls.NOTEBOOKS,
            cls.TESTS,
            
            # Cache
            cls.CACHE,
            cls.TEMP,
            
            # Docs
            cls.DOCS,
        ]
        
        created = []
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory)
        
        return created
    
    @classmethod
    def verify_structure(cls) -> dict:
        """Verify that all critical directories exist."""
        status = {}
        
        critical_dirs = [
            ("Project Root", cls.PROJECT_ROOT),
            ("Data Root", cls.DATA_ROOT),
            ("Raw Wiki File", cls.RAW_WIKI_FILE.parent),
            ("Source Code", cls.SRC),
            ("Models", cls.MODELS),
            ("Logs", cls.LOGS),
        ]
        
        for name, path in critical_dirs:
            status[name] = path.exists()
        
        return status


# ============================================================================
# GLOBAL PATHS INSTANCE
# ============================================================================

PATHS = Paths()


# ============================================================================
# PATH UTILITIES
# ============================================================================

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoint_path(stage_name: str, step: int) -> Path:
    """Generate checkpoint path for a given stage and step."""
    checkpoint_dir = PATHS.get_stage_checkpoint_dir(stage_name)
    ensure_dir(checkpoint_dir)
    return checkpoint_dir / f"checkpoint_step_{step}.pt"


def get_latest_checkpoint(stage_name: str) -> Path:
    """Find the latest checkpoint for a given stage."""
    checkpoint_dir = PATHS.get_stage_checkpoint_dir(stage_name)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoints[-1]


def get_best_model_path(stage_name: str) -> Path:
    """Get path for the best model of a stage."""
    return PATHS.BEST_MODELS / f"{stage_name}_best.pt"


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_paths():
    """Validate that all critical paths are accessible."""
    print("="*80)
    print("PATH VALIDATION")
    print("="*80)
    
    status = PATHS.verify_structure()
    
    all_good = True
    for name, exists in status.items():
        symbol = "✓" if exists else "✗"
        status_text = "EXISTS" if exists else "MISSING"
        print(f"{symbol} {name:.<50} {status_text}")
        
        if not exists:
            all_good = False
    
    print("="*80)
    
    if all_good:
        print("✓ All critical paths are accessible!")
    else:
        print("✗ Some paths are missing. Run initialize_project() to create them.")
    
    return all_good


def initialize_project():
    """Initialize the entire project structure."""
    print("="*80)
    print("INITIALIZING WIKIFORGE-GPT PROJECT")
    print("="*80)
    print(f"\nProject Root: {PATHS.PROJECT_ROOT}")
    print(f"Data Root: {PATHS.DATA_ROOT}")
    print(f"\nCreating directory structure...\n")
    
    created = PATHS.create_all()
    
    if created:
        print(f"✓ Created {len(created)} directories:")
        for directory in created:
            print(f"  - {directory}")
    else:
        print("✓ All directories already exist")
    
    print("\n" + "="*80)
    print("✓ PROJECT INITIALIZATION COMPLETE")
    print("="*80)
    
    # Validate
    print()
    validate_paths()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # When run directly, initialize and validate
    initialize_project()
