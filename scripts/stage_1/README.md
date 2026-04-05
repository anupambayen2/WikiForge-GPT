# STAGE_1 - 45M Parameters

## Overview
- **Parameters**: 45M
- **Architecture**: 8 layers, 8 heads, 512 d_model
- **Vocabulary**: 16,000 tokens
- **Training time**: 14 hours
- **Expected loss**: ~2.5

## Quick Start

```bash
# Step 1: Train tokenizer (~15-25 min)
python scripts/stage_1/step1_train_tokenizer.py

# Step 2: Tokenize dataset (~45-75 min)
python scripts/stage_1/step2_tokenize_dataset.py

# Step 3: Train model (14 hours)
python scripts/stage_1/step3_train_model.py
```

## Enhanced Features

✅ **Time Tracking**
- Elapsed time display
- ETA (time remaining) calculation
- Steps per second

✅ **Perplexity Calculation**
- Computed every 1000 steps
- Logged alongside validation loss
- Tracks best perplexity

✅ **Detailed Logging**
- JSONL log file with all metrics
- Timestamp for each entry
- Both training and evaluation logs

## Monitoring Training

Watch for these metrics:
- **Loss**: Should decrease from ~70 → ~2.5
- **Perplexity**: Should decrease (lower is better)
- **Steps/sec**: Indicates training speed
- **ETA**: Time remaining

## Output Files

Model checkpoints:
```
E:/WikiForge-GPT/models/checkpoints/stage_1/checkpoints/
├── checkpoint_step_5000.pt
├── checkpoint_step_10000.pt
├── ...
├── checkpoint_step_100000.pt
└── best_model.pt  ⭐ (best validation loss)
```

Training logs:
```
E:/WikiForge-GPT/logs/training/
└── stage_1_training_log.jsonl
```

## Next Steps

After completing this stage, proceed to the next stage folder for continued training!
