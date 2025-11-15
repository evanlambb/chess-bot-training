# Chess Dataset Generator

Generate high-quality chess training datasets using Stockfish self-play for training neural network chess engines.

## Overview

This dataset generator creates training data specifically for **engine-vs-engine** competition by:
1. ✅ Generating positions through Stockfish self-play
2. ✅ Labeling positions with Stockfish evaluations
3. ✅ Encoding boards in neural-network-friendly format
4. ✅ Supporting parallel generation for cloud compute
5. ✅ Providing ready-to-use PyTorch data loaders

## Installation

### 1. Install Python Dependencies

```bash
cd dataset_generator
pip install -r requirements.txt
```

### 2. Install Stockfish

#### Windows:
1. Download from: https://stockfishchess.org/download/
2. Extract `stockfish.exe` to a known location (e.g., `C:\stockfish\stockfish.exe`)
3. Update `stockfish_path` in the config

#### macOS:
```bash
brew install stockfish
```

#### Linux:
```bash
sudo apt-get install stockfish
# or
sudo yum install stockfish
```

### 3. Verify Stockfish Installation

```bash
# Windows
C:\stockfish\stockfish.exe

# macOS/Linux
stockfish
```

You should see Stockfish start. Type `quit` to exit.

## Quick Start

### Generate a Small Test Dataset

```python
python generate_dataset.py
```

This will generate 100 games (~5,000 positions) using default settings.

### Customize Generation

Edit the config in `generate_dataset.py`:

```python
config = DatasetConfig(
    # Stockfish path (UPDATE THIS!)
    stockfish_path="stockfish",  # or full path like "C:\\stockfish\\stockfish.exe"
    
    # Performance
    threads=1,
    hash_mb=256,
    
    # Search strength
    play_depth=10,      # Depth for generating moves
    label_depth=12,     # Depth for labeling (can be deeper)
    
    # Alternative: use time instead of depth
    # play_time_ms=20,
    # label_time_ms=50,
    
    # Dataset size
    num_games=1000,     # Number of games to generate
    max_moves=100,      # Max moves per game (100 = 200 plies)
    
    # Variety
    opening_variety=True,   # Add random opening moves
    opening_moves=4,        # Number of random opening moves
    
    # Output
    output_dir="chess_dataset",
    save_every=100,     # Checkpoint frequency
    
    # Parallel processing (IMPORTANT FOR CLOUD!)
    parallel_workers=1,  # Set to 4-16 for cloud compute
)
```

## Cloud Compute

For faster generation, use **parallel workers**:

```python
config = DatasetConfig(
    parallel_workers=8,  # Use 8 parallel Stockfish instances
    num_games=10000,     # Generate 10k games
    # ... other settings
)
```

**Recommended cloud settings:**
- 16-32 CPU cores
- `parallel_workers = num_cores - 2`
- `threads = 1` (per Stockfish instance)
- Total compute = `parallel_workers * threads`

Example on 16-core machine:
```python
parallel_workers=14  # Leave 2 cores for system
threads=1
```

## Data Format

The generated dataset is saved as compressed NPZ files with:

### Arrays:
- **`boards`**: `(N, 14, 8, 8)` - Encoded board positions
  - Planes 0-5: Current player's pieces (P, N, B, R, Q, K)
  - Planes 6-11: Opponent's pieces (P, N, B, R, Q, K)
  - Plane 12: Castling rights
  - Plane 13: En passant, turn, move count
  
- **`policy_indices`**: `(N,)` - Best move indices (from_square * 64 + to_square)
- **`policy_ucis`**: `(N,)` - Best moves in UCI format (e.g., "e2e4")
- **`values`**: `(N,)` - Position evaluations in [-1, 1]
- **`fens`**: `(N,)` - FEN strings (for debugging)
- **`game_ids`**: `(N,)` - Game IDs

## Usage in Training

### Load Dataset

```python
from data_loader import ChessDataset, create_data_loader

# Simple loading
dataset = ChessDataset("chess_dataset/dataset_final.npz")

# With DataLoader
loader = create_data_loader(
    "chess_dataset/dataset_final.npz",
    batch_size=256,
    shuffle=True,
    num_workers=4
)

# Iterate
for boards, policies, values in loader:
    # boards: (batch, 14, 8, 8)
    # policies: (batch,) - move indices
    # values: (batch,) - position evals
    train_step(boards, policies, values)
```

### Example Training

```python
python example_training.py
```

See `example_training.py` for a complete training example with:
- Simple CNN architecture
- Dual-head (policy + value)
- Training loop with validation

## File Structure

```
dataset_generator/
├── generate_dataset.py      # Main dataset generator
├── data_loader.py            # PyTorch dataset & utilities
├── example_training.py       # Example training script
├── requirements.txt          # Python dependencies
└── README.md                # This file

chess_dataset/               # Generated data (created automatically)
├── dataset_final.npz        # Final dataset
├── dataset_checkpoint_100.npz  # Checkpoints
├── stats_final.json         # Dataset statistics
└── config.json              # Generation config
```

## Configuration Guide

### v1 (Simple): Same depth for play and label

```python
play_depth=10,
label_depth=10,
```

- Fast
- Good for initial testing
- Adequate quality

### v2 (Better): Separate budgets

```python
play_depth=8,    # Fast, generates positions quickly
label_depth=14,  # Deeper, better labels
```

- Positions from ~2000 ELO play
- Labels from ~2800 ELO analysis
- Better data quality

### v3 (Time-based): Consistent hardware performance

```python
play_time_ms=10,   # 10ms per move during play
label_time_ms=50,  # 50ms for labeling
```

- More consistent across different hardware
- Easier to budget time

## Performance Tips

1. **Start small**: Test with 100 games first
2. **Monitor progress**: Logs show games/sec and ETA
3. **Use checkpoints**: Set `save_every=100` to avoid data loss
4. **Cloud scaling**: Use `parallel_workers` for massive speedup
5. **Storage**: ~1KB per position, so 500k positions ≈ 500MB

## Expected Generation Times

With depth 10/12 on modern CPU:

| Workers | Games | Positions | Time |
|---------|-------|-----------|------|
| 1 | 100 | ~5k | ~15 min |
| 1 | 1,000 | ~50k | ~2.5 hours |
| 8 | 1,000 | ~50k | ~20 min |
| 8 | 10,000 | ~500k | ~3 hours |

## Troubleshooting

### "Stockfish not found"
- Update `stockfish_path` to full path
- Windows: `r"C:\stockfish\stockfish.exe"`
- Verify Stockfish runs from command line

### Out of memory
- Reduce `parallel_workers`
- Reduce `hash_mb`
- Reduce `save_every` to checkpoint more often

### Slow generation
- Increase `parallel_workers`
- Reduce search depth
- Use time-based search

## Next Steps

1. **Generate test dataset**: 100 games to verify everything works
2. **Scale up**: Generate 10k-100k games for real training
3. **Train model**: Use `example_training.py` as starting point
4. **Iterate**: Add your model's own games to training data
5. **Improve**: Generate new data from your model vs Stockfish

## Advanced: Self-Play Loop

Once you have a trained model, generate data from your model playing itself or playing Stockfish:

```python
# TODO: Add your model self-play here
# This creates a feedback loop for continuous improvement
```

## Questions?

This generator is designed to be:
- ✅ Simple to use
- ✅ Production-ready
- ✅ Scalable to cloud
- ✅ Compatible with python-chess

Adjust the config to match your needs and hardware!

