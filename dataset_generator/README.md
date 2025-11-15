# Chess Dataset Generator

This directory contains tools for generating chess training datasets using Stockfish self-play.

## Quick Start

### Generate Locally (Sequential)
```bash
python generate_dataset.py
```

### Generate on Modal (Parallel, Fast)
```bash
# Setup (one-time)
pip install modal
modal setup

# Generate 1,000 games (~3-5 minutes)
modal run generate_dataset_modal.py --num-games 1000 --parallel-workers 30 --dataset-name chess_1k

# Download from Modal
modal volume get chess-datasets chess_1k/dataset_final.npz ./chess_dataset/
```

See [MODAL_GUIDE.md](MODAL_GUIDE.md) for detailed Modal instructions.

### Merge Datasets
```bash
python merge_datasets.py \
    chess_dataset/dataset_final.npz \
    chess_dataset/chess_1k_dataset.npz \
    -o chess_dataset_merged/dataset_final.npz
```

## Files

### Core Scripts
- **`generate_dataset.py`** - Local dataset generation (sequential)
- **`generate_dataset_modal.py`** - Modal cloud generation (parallel, 100x faster)
- **`merge_datasets.py`** - Utility to combine multiple datasets

### Documentation
- **`MODAL_GUIDE.md`** - Complete guide for using Modal cloud compute
- **`README.md`** - This file

### Configuration
- **`requirements.txt`** - Python dependencies

### Data
- **`chess_dataset/`** - Generated datasets
  - `dataset_final.npz` - Original dataset (~300 games)
  - `chess_1k_dataset.npz` - Modal-generated dataset (1000 games)
  - `chess_1.3k.npz` - Merged dataset (1300 games)
  - `config.json` - Generation configuration
  - `stats_*.json` - Dataset statistics

## Dataset Format

Each `.npz` file contains:
- `boards` - (N, 14, 8, 8) board encodings
- `policy_indices` - (N,) move targets (from_square * 64 + to_square)
- `policy_ucis` - (N,) moves in UCI format
- `values` - (N,) position evaluations in [-1, 1]
- `fens` - (N,) FEN strings for debugging
- `game_ids` - (N,) game identifiers

## Key Features

✅ **Improved Diversity** - 10 random opening moves + stochastic play (15% of moves)
✅ **Numerically Stable** - Fixed softmax overflow issues
✅ **Scalable** - Modal integration for 50-100+ parallel workers
✅ **Fast** - Generate 1000 games in 3-5 minutes (Modal) vs hours (local)

## Training

To train your model with the generated data, use:
```bash
python ../src/main.py --dataset chess_dataset/chess_1.3k.npz
```

See `../src/main.py` for the actual training implementation.
