"""
WikiForge-GPT Project Settings
================================
Enterprise-level configuration for building GPT from scratch using Wikipedia data.

Author: Anupam
Project: WikiForge-GPT
Hardware: RTX 4060 8GB VRAM, i5-12450H, 16GB RAM
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================================
# PROJECT PATHS CONFIGURATION
# ============================================================================

@dataclass
class ProjectPaths:
    """Centralized path management for the entire project."""
    
    # Root directories
    PROJECT_ROOT: Path = Path("E:/WikiForge-GPT")
    DATA_ROOT: Path = Path("F:/wiki_raw")
    
    # Data directories
    RAW_DATA: Path = field(init=False)
    PROCESSED_DATA: Path = field(init=False)
    TOKENIZER_DATA: Path = field(init=False)
    TRAINING_DATA: Path = field(init=False)
    VALIDATION_DATA: Path = field(init=False)
    
    # Model directories
    MODELS: Path = field(init=False)
    CHECKPOINTS: Path = field(init=False)
    BEST_MODELS: Path = field(init=False)
    
    # Training artifacts
    LOGS: Path = field(init=False)
    TENSORBOARD: Path = field(init=False)
    METRICS: Path = field(init=False)
    
    # Output directories
    OUTPUTS: Path = field(init=False)
    PREDICTIONS: Path = field(init=False)
    SAMPLES: Path = field(init=False)
    
    # Utilities
    CONFIGS: Path = field(init=False)
    SCRIPTS: Path = field(init=False)
    NOTEBOOKS: Path = field(init=False)
    CACHE: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize all subdirectories."""
        # Data directories
        self.RAW_DATA = self.DATA_ROOT / "raw"
        self.PROCESSED_DATA = self.PROJECT_ROOT / "data" / "processed"
        self.TOKENIZER_DATA = self.PROJECT_ROOT / "data" / "tokenizer"
        self.TRAINING_DATA = self.PROJECT_ROOT / "data" / "training"
        self.VALIDATION_DATA = self.PROJECT_ROOT / "data" / "validation"
        
        # Model directories
        self.MODELS = self.PROJECT_ROOT / "models"
        self.CHECKPOINTS = self.MODELS / "checkpoints"
        self.BEST_MODELS = self.MODELS / "best"
        
        # Training artifacts
        self.LOGS = self.PROJECT_ROOT / "logs"
        self.TENSORBOARD = self.LOGS / "tensorboard"
        self.METRICS = self.LOGS / "metrics"
        
        # Output directories
        self.OUTPUTS = self.PROJECT_ROOT / "outputs"
        self.PREDICTIONS = self.OUTPUTS / "predictions"
        self.SAMPLES = self.OUTPUTS / "samples"
        
        # Utilities
        self.CONFIGS = self.PROJECT_ROOT / "configs"
        self.SCRIPTS = self.PROJECT_ROOT / "scripts"
        self.NOTEBOOKS = self.PROJECT_ROOT / "notebooks"
        self.CACHE = self.PROJECT_ROOT / ".cache"
    
    def create_all_directories(self):
        """Create all project directories if they don't exist."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Path) and not attr_name.startswith('_'):
                attr.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created: {attr}")


# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

@dataclass
class HardwareConfig:
    """Hardware specifications and constraints."""
    
    # GPU Configuration
    GPU_NAME: str = "NVIDIA GeForce RTX 4060 Laptop"
    VRAM_GB: int = 8
    CUDA_VERSION: str = "12.6"
    
    # CPU Configuration
    CPU_NAME: str = "12th Gen Intel(R) Core(TM) i5-12450H"
    CPU_CORES: int = 12  # 4 P-cores + 8 E-cores
    RAM_GB: float = 15.7
    
    # Training constraints based on hardware
    MAX_BATCH_SIZE: int = 4  # Conservative for 8GB VRAM
    GRADIENT_ACCUMULATION_STEPS: int = 8  # Effective batch size = 32
    USE_MIXED_PRECISION: bool = True  # FP16/BF16 to save memory
    USE_GRADIENT_CHECKPOINTING: bool = True  # Trade compute for memory
    
    # DataLoader configuration
    NUM_WORKERS: int = 4  # Conservative for 12 cores
    PIN_MEMORY: bool = True


# ============================================================================
# MODEL ARCHITECTURE STAGES
# ============================================================================

@dataclass
class ModelStageConfig:
    """Configuration for incremental model training stages."""
    
    stage_name: str
    vocab_size: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    max_seq_length: int
    dropout: float
    estimated_params_millions: float
    estimated_vram_gb: float
    training_hours: int
    description: str


# Stage configurations - incremental approach
TRAINING_STAGES = [
    ModelStageConfig(
        stage_name="stage_0_tiny",
        vocab_size=8000,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        max_seq_length=256,
        dropout=0.1,
        estimated_params_millions=12,
        estimated_vram_gb=2.5,
        training_hours=48,
        description="Tiny model - proof of concept, learn the pipeline"
    ),
    ModelStageConfig(
        stage_name="stage_1_small",
        vocab_size=16000,
        n_layers=6,
        n_heads=6,
        d_model=384,
        d_ff=1536,
        max_seq_length=512,
        dropout=0.1,
        estimated_params_millions=45,
        estimated_vram_gb=4.0,
        training_hours=96,
        description="Small model - validate architecture and training loop"
    ),
    ModelStageConfig(
        stage_name="stage_2_medium",
        vocab_size=32000,
        n_layers=8,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        max_seq_length=1024,
        dropout=0.1,
        estimated_params_millions=125,
        estimated_vram_gb=6.5,
        training_hours=168,
        description="Medium model - approaching GPT-2 small size"
    ),
    ModelStageConfig(
        stage_name="stage_3_large",
        vocab_size=50000,
        n_layers=12,
        n_heads=12,
        d_model=768,
        d_ff=3072,
        max_seq_length=1024,
        dropout=0.1,
        estimated_params_millions=350,
        estimated_vram_gb=7.8,
        training_hours=336,
        description="Large model - GPT-2 medium equivalent (max for 8GB VRAM)"
    ),
]


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """General training hyperparameters."""
    
    # Optimizer settings
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 0.1
    BETA1: float = 0.9
    BETA2: float = 0.95
    EPSILON: float = 1e-8
    
    # Learning rate schedule
    WARMUP_STEPS: int = 2000
    LR_DECAY_STYLE: str = "cosine"
    MIN_LR_RATIO: float = 0.1
    
    # Regularization
    GRAD_CLIP_NORM: float = 1.0
    LABEL_SMOOTHING: float = 0.0
    
    # Checkpointing
    SAVE_INTERVAL_HOURS: int = 6
    KEEP_LAST_N_CHECKPOINTS: int = 5
    SAVE_BEST_MODEL: bool = True
    
    # Validation
    VALIDATION_INTERVAL_STEPS: int = 1000
    VALIDATION_SAMPLES: int = 1000
    
    # Logging
    LOG_INTERVAL_STEPS: int = 100
    SAMPLE_GENERATION_INTERVAL: int = 5000
    
    # Safety
    MAX_TRAINING_HOURS: int = 48  # Stop after 48 hours
    AUTO_RESUME: bool = True  # Resume from checkpoint if exists


# ============================================================================
# DATA PROCESSING CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Configuration for data processing pipeline."""
    
    # Wikipedia XML processing
    XML_FILENAME: str = "enwiki-latest-pages-articles.xml_2"
    CHUNK_SIZE_MB: int = 500  # Process XML in chunks
    
    # Text cleaning
    MIN_ARTICLE_LENGTH: int = 100  # Minimum characters
    MAX_ARTICLE_LENGTH: int = 50000  # Maximum characters
    REMOVE_SPECIAL_PAGES: bool = True
    REMOVE_DISAMBIGUATION: bool = True
    
    # Tokenizer training
    TOKENIZER_TYPE: str = "BPE"  # Byte-Pair Encoding
    TOKENIZER_TRAINING_SAMPLES: int = 1000000  # 1M articles for tokenizer
    
    # Data splits
    TRAIN_SPLIT: float = 0.95
    VAL_SPLIT: float = 0.04
    TEST_SPLIT: float = 0.01
    
    # Sequence preparation
    STRIDE: int = 128  # Overlapping window for long documents


# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig:
    """System-level configuration."""
    
    # Random seeds for reproducibility
    RANDOM_SEED: int = 42
    
    # Device configuration
    DEVICE: str = "cuda"  # Will auto-detect
    
    # Memory management
    EMPTY_CACHE_INTERVAL: int = 100  # Clear CUDA cache every N steps
    
    # Multiprocessing
    MP_START_METHOD: str = "spawn"  # For Windows compatibility
    
    # Monitoring
    MONITOR_GPU_MEMORY: bool = True
    MONITOR_CPU_MEMORY: bool = True
    
    # Safety checks
    ENABLE_ANOMALY_DETECTION: bool = False  # Debugging mode (slower)
    CHECK_GRADIENTS: bool = True


# ============================================================================
# INSTANTIATE CONFIGURATIONS
# ============================================================================

# Create global configuration instances
PATHS = ProjectPaths()
HARDWARE = HardwareConfig()
TRAINING = TrainingConfig()
DATA = DataConfig()
SYSTEM = SystemConfig()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def initialize_project():
    """Initialize the project by creating all necessary directories."""
    print("="*80)
    print("WikiForge-GPT Project Initialization")
    print("="*80)
    print(f"\nProject Root: {PATHS.PROJECT_ROOT}")
    print(f"Data Root: {PATHS.DATA_ROOT}")
    print(f"\nHardware: {HARDWARE.GPU_NAME} ({HARDWARE.VRAM_GB}GB VRAM)")
    print(f"CPU: {HARDWARE.CPU_NAME}")
    print("\nCreating directory structure...")
    print("-"*80)
    
    PATHS.create_all_directories()
    
    print("-"*80)
    print("✓ Project initialization complete!")
    print("="*80)


def get_stage_config(stage_name: str) -> Optional[ModelStageConfig]:
    """Get configuration for a specific training stage."""
    for stage in TRAINING_STAGES:
        if stage.stage_name == stage_name:
            return stage
    return None


def print_stage_summary():
    """Print summary of all training stages."""
    print("\n" + "="*80)
    print("TRAINING STAGES OVERVIEW")
    print("="*80)
    
    for i, stage in enumerate(TRAINING_STAGES):
        print(f"\nStage {i}: {stage.stage_name.upper()}")
        print(f"  Parameters: {stage.estimated_params_millions}M")
        print(f"  VRAM Required: {stage.estimated_vram_gb}GB")
        print(f"  Training Time: {stage.training_hours}h ({stage.training_hours/24:.1f} days)")
        print(f"  Vocab Size: {stage.vocab_size:,}")
        print(f"  Layers: {stage.n_layers} | Heads: {stage.n_heads} | D-model: {stage.d_model}")
        print(f"  Max Sequence: {stage.max_seq_length} tokens")
        print(f"  Description: {stage.description}")
    
    total_days = sum(s.training_hours for s in TRAINING_STAGES) / 24
    print(f"\n{'='*80}")
    print(f"Total Estimated Training Time: {total_days:.1f} days")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    initialize_project()
    print_stage_summary()
    
    print("\nNext Steps:")
    print("1. Run this script to create directory structure")
    print("2. Review paths.py for centralized path management")
    print("3. Follow the project plan in PROJECT_PLAN.md")
    print("4. Start with Stage 0 (Tiny model) to validate pipeline")
