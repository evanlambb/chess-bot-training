# Chess Dataset Generator - Project Overview

A production-ready system for generating high-quality chess training datasets using Stockfish self-play, specifically designed for training neural network chess engines that compete against other engines.

## 🎯 Purpose

Generate training data that matches the **engine-vs-engine meta**, not human play. This ensures your neural network learns to handle:
- Tactically sharp positions
- Engine-quality play
- Subtle mistakes engines make (not human blunders)

## 📦 What's Included

### Core Files

| File | Purpose |
|------|---------|
| `generate_dataset.py` | Main dataset generator with Stockfish self-play |
| `data_loader.py` | PyTorch dataset utilities for loading/using data |
| `example_training.py` | Example neural network training script |
| `config_template.py` | Pre-configured templates for different use cases |
| `quick_start.py` | Interactive setup wizard for beginners |
| `visualize_data.py` | Inspect and visualize generated datasets |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Complete usage guide |
| `INSTALL.md` | Detailed installation instructions |
| `PROJECT_OVERVIEW.md` | This file - high-level overview |

### Setup Scripts

| File | Purpose |
|------|---------|
| `setup.bat` | Windows setup script |
| `setup.sh` | macOS/Linux setup script |
| `requirements.txt` | Python dependencies |

## 🚀 Quick Start (3 Steps)

### 1. Install
```bash
# Windows
setup.bat

# macOS/Linux
chmod +x setup.sh && ./setup.sh
```

### 2. Generate
```bash
python quick_start.py
```

### 3. Train
```bash
python example_training.py
```

## 🏗️ Architecture

### Data Generation Pipeline

```
┌─────────────────┐
│  Stockfish      │
│  Self-Play      │  Generate engine-quality games
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Position       │
│  Extraction     │  Extract positions from games
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stockfish      │
│  Labeling       │  Deep analysis for quality labels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Board          │
│  Encoding       │  Convert to neural network format
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NPZ Dataset    │  Compressed numpy arrays
└─────────────────┘
```

### Board Encoding Format

**14 planes of 8×8:**
- Planes 0-5: Current player's pieces (P, N, B, R, Q, K)
- Planes 6-11: Opponent's pieces (P, N, B, R, Q, K)
- Plane 12: Castling rights
- Plane 13: En passant, turn, move count

Shape: `(14, 8, 8)` per position

### Move Encoding

Two formats provided:
1. **Policy index**: `from_square * 64 + to_square` (0-4095)
2. **UCI string**: `"e2e4"`, `"e7e8q"`, etc.

### Value Encoding

Position evaluation in `[-1, 1]`:
- +1.0: Winning for current player
- 0.0: Equal
- -1.0: Losing for current player

Converted from centipawn scores using clipping and normalization.

## 📊 Dataset Sizes

| Size | Games | Positions | Time (1 core) | Time (8 cores) | Use Case |
|------|-------|-----------|---------------|----------------|----------|
| Tiny | 10 | ~500 | 2 min | <1 min | Testing |
| Small | 100 | ~5k | 15 min | 2 min | Prototyping |
| Medium | 1,000 | ~50k | 2.5 hours | 20 min | Development |
| Large | 10,000 | ~500k | 25 hours | 3 hours | Production |
| Huge | 100,000 | ~5M | 250 hours | 30 hours | SOTA |

## ⚙️ Configuration Guide

### Three Configuration Strategies

#### v1: Simple (same depth)
```python
play_depth=10,
label_depth=10,
```
- Fast, good for testing
- Play and labels have same quality

#### v2: Two-stage (recommended)
```python
play_depth=8,    # ~2000 ELO play
label_depth=14,  # ~2800 ELO labels
```
- Best balance
- Fast game generation
- High-quality labels

#### v3: Time-based
```python
play_time_ms=10,   # 10ms per move
label_time_ms=50,  # 50ms for analysis
```
- Hardware-independent
- Consistent across systems

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stockfish_path` | `"stockfish"` | Path to Stockfish executable |
| `play_depth` | 10 | Search depth during self-play |
| `label_depth` | 12 | Search depth for labeling |
| `num_games` | 1000 | Number of games to generate |
| `parallel_workers` | 1 | CPU cores to use (increase for cloud!) |
| `opening_variety` | True | Use random openings |
| `opening_moves` | 4 | Number of random opening moves |

## 🔧 Advanced Features

### Parallel Generation

Built-in support for multi-core generation:

```python
config = DatasetConfig(
    parallel_workers=8,  # Use 8 CPU cores
    # Each worker runs independent Stockfish
)
```

**Cloud recommendations:**
- Use `workers = total_cores - 2`
- Keep `threads = 1` per worker
- Best on 16+ core machines

### Checkpointing

Automatic checkpoints during generation:

```python
config = DatasetConfig(
    save_every=100,  # Save every 100 games
)
```

Creates:
- `dataset_checkpoint_100.npz`
- `dataset_checkpoint_200.npz`
- etc.

### Merging Datasets

Combine multiple datasets:

```python
from data_loader import merge_datasets

merge_datasets(
    ['dataset1.npz', 'dataset2.npz', 'dataset3.npz'],
    'merged_dataset.npz'
)
```

## 📈 Training Integration

### PyTorch Dataset

```python
from data_loader import ChessDataset

dataset = ChessDataset("chess_dataset/dataset_final.npz")

# Get a sample
board, policy, value = dataset[0]
# board: (14, 8, 8) tensor
# policy: scalar (move index)
# value: scalar (evaluation)
```

### DataLoader

```python
from data_loader import create_data_loader

loader = create_data_loader(
    "chess_dataset/dataset_final.npz",
    batch_size=256,
    shuffle=True,
    num_workers=4
)

for boards, policies, values in loader:
    # Train your model
    pass
```

## 🎓 Example Neural Network

Included example with:
- **ResNet-style architecture** (convolutional + residual blocks)
- **Dual-head output** (policy + value)
- **Complete training loop** with validation
- **Checkpointing** and learning rate scheduling

See `example_training.py` for details.

## 🛠️ Utilities

### Visualize Dataset

```bash
python visualize_data.py dataset_final.npz
```

Shows:
- Dataset statistics
- Value distribution
- Move distribution
- Sample positions with boards

### Compare Datasets

```bash
python visualize_data.py compare dataset1.npz dataset2.npz
```

### Inspect Sample

```python
from data_loader import ChessDataset

dataset = ChessDataset("dataset.npz")
sample = dataset.get_sample_with_metadata(0)

print(sample['fen'])
print(sample['policy_uci'])
print(sample['value'])
```

## 📁 Output Structure

After generation:

```
chess_dataset/
├── dataset_final.npz           # Main dataset (compressed)
├── dataset_checkpoint_100.npz  # Checkpoint 1
├── dataset_checkpoint_200.npz  # Checkpoint 2
├── stats_final.json            # Dataset statistics
├── stats_checkpoint_100.json   # Checkpoint stats
└── config.json                 # Generation configuration
```

## 🔍 Quality Assurance

The generator includes:

1. **Automatic opening variety** - Prevents overfitting to single openings
2. **Move count limits** - Prevents ultra-long games
3. **Game result tracking** - Ensures diverse outcomes
4. **Value clipping** - Prevents extreme mate scores from skewing training
5. **Metadata preservation** - FEN strings for debugging

## 🌩️ Cloud Deployment

### Recommended Setup

```python
# On 16-core cloud instance
config = DatasetConfig(
    parallel_workers=14,  # Leave 2 for system
    num_games=10000,
    hash_mb=512,
)
```

### Run in Background

```bash
# Start generation
nohup python generate_dataset.py > generation.log 2>&1 &

# Monitor progress
tail -f generation.log

# Check if still running
ps aux | grep generate_dataset
```

### Download Results

```bash
# Compress for download
tar -czf dataset.tar.gz chess_dataset/

# Download via SCP
scp user@instance:~/dataset.tar.gz .
```

## 🎯 Best Practices

### For ChessHacks (36-hour hackathon)

```python
config = DatasetConfig(
    play_depth=8,
    label_depth=12,
    num_games=5000,      # ~250k positions
    parallel_workers=12,  # Use cloud compute
)
```

**Timeline:**
- Hour 0-2: Setup + test generation
- Hour 2-6: Generate dataset on cloud
- Hour 6-36: Train and iterate model

### For Production

```python
# Phase 1: Initial training
config = DatasetConfig(
    num_games=10000,  # ~500k positions
    parallel_workers=16,
)

# Phase 2: Add your model's self-play
# Generate games from your model vs Stockfish
# Mix with original dataset

# Phase 3: Continuous improvement
# Periodically generate new data with updated model
```

## 🔄 Improvement Loop

1. **Generate** initial dataset (Stockfish self-play)
2. **Train** your neural network
3. **Evaluate** against Stockfish
4. **Generate** new data (your model vs Stockfish)
5. **Merge** with original dataset
6. **Retrain** with combined data
7. Repeat steps 3-6

## 📊 Performance Tips

### Speed Up Generation
- Increase `parallel_workers`
- Reduce `play_depth`
- Use time-based search
- Use cloud with many cores

### Improve Quality
- Increase `label_depth`
- More `opening_moves` for variety
- Generate more games
- Use stronger Stockfish version

### Save Resources
- Reduce `hash_mb`
- Lower `parallel_workers`
- Generate in batches
- Use checkpointing

## 🐛 Debugging

### Verbose Mode

Edit `generate_dataset.py`:
```python
logging.basicConfig(level=logging.DEBUG)  # Was INFO
```

### Test Single Game

```python
from generate_dataset import generate_single_game, DatasetConfig

config = DatasetConfig(num_games=1)
samples = generate_single_game(config, 0, "stockfish")
print(f"Generated {len(samples)} positions")
```

### Verify Stockfish

```python
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("stockfish")
print(engine.id)
engine.quit()
```

## 📚 Further Reading

- **AlphaZero paper**: Similar architecture concepts
- **Stockfish docs**: Understanding engine parameters
- **python-chess docs**: Board manipulation
- **PyTorch tutorials**: Neural network training

## 🤝 Integration with Your Bot

To use this with your existing bot (`my-chesshacks-bot/src/main.py`):

1. **Generate dataset** using this generator
2. **Train model** using `example_training.py` as template
3. **Load model** in your bot's setup:
   ```python
   import torch
   from your_model import ChessNet
   
   model = ChessNet()
   model.load_state_dict(torch.load('chess_model_final.pt'))
   model.eval()
   ```
4. **Use in entrypoint**:
   ```python
   @chess_manager.entrypoint
   def test_func(ctx: GameContext):
       board_tensor = encode_board(ctx.board)
       policy, value = model(board_tensor)
       # Pick move from policy distribution
       return best_move
   ```

## 🎉 Summary

This dataset generator provides:

✅ **Engine-quality training data**  
✅ **Flexible configuration**  
✅ **Scalable to cloud compute**  
✅ **Production-ready code**  
✅ **Complete training pipeline**  
✅ **Extensive documentation**

Perfect for ChessHacks, research, or building competitive chess engines!

---

**Questions?** Check `README.md` and `INSTALL.md` for detailed guides.

**Issues?** See troubleshooting sections in documentation.

**Ready?** Run `python quick_start.py` to begin! 🚀

