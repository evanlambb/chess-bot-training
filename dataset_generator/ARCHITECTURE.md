# System Architecture

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Chess Dataset Generator                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│   Configuration     │
│  (config_template)  │
│                     │
│  • play_depth       │
│  • label_depth      │
│  • num_games        │
│  • parallel_workers │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Game Generation Loop                          │
│                  (generate_dataset.py)                           │
└─────────────────────────────────────────────────────────────────┘
           │
           ├─────────────┬─────────────┬─────────────┐
           │             │             │             │
           ▼             ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │
    │          │  │          │  │          │  │          │
    │Stockfish │  │Stockfish │  │Stockfish │  │Stockfish │
    │Self-Play │  │Self-Play │  │Self-Play │  │Self-Play │
    └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
         │             │             │             │
         └─────────────┴─────────────┴─────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Position Collection   │
              │                        │
              │  For each position:    │
              │  1. Play move (depth)  │
              │  2. Label (depth+2)    │
              │  3. Encode board       │
              │  4. Encode move        │
              │  5. Convert value      │
              └────────┬───────────────┘
                       │
                       ▼
              ┌────────────────────────┐
              │    Data Encoding       │
              │   (board → tensor)     │
              │                        │
              │  • 14 planes 8×8       │
              │  • Policy indices      │
              │  • Value normalization │
              └────────┬───────────────┘
                       │
                       ▼
              ┌────────────────────────┐
              │   Save to NPZ          │
              │   (compressed)         │
              │                        │
              │  • boards.npy          │
              │  • policy_indices.npy  │
              │  • values.npy          │
              │  • metadata            │
              └────────┬───────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Dataset Files                             │
│                                                                  │
│  chess_dataset/                                                  │
│  ├── dataset_final.npz         ← Main dataset                   │
│  ├── dataset_checkpoint_N.npz  ← Checkpoints                    │
│  ├── stats_final.json          ← Statistics                     │
│  └── config.json               ← Generation config              │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Data Loading                                │
│                    (data_loader.py)                              │
│                                                                  │
│  ChessDataset(npz_path)                                          │
│  ├── __len__()                                                   │
│  ├── __getitem__(idx) → (board, policy, value)                  │
│  └── get_sample_with_metadata(idx)                              │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Training Loop                              │
│                  (example_training.py)                           │
│                                                                  │
│  for epoch in epochs:                                            │
│      for boards, policies, values in dataloader:                 │
│          policy_logits, value_pred = model(boards)               │
│          loss = policy_loss + value_loss                         │
│          optimizer.step()                                        │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
              ┌────────────────┐
              │  Trained Model │
              │                │
              │  chess_model   │
              │  _final.pt     │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────────┐
              │  Integration with  │
              │  Your Chess Bot    │
              │                    │
              │  (main.py)         │
              └────────────────────┘
```

## Data Flow Detail

### 1. Position Encoding

```
Chess Board (FEN)
   ↓
┌─────────────────────────────────────┐
│  rnbqkbnr/pppppppp/8/8/8/8/...     │  FEN String
└─────────────────────────────────────┘
   ↓  encode_board()
┌─────────────────────────────────────┐
│  Tensor (14, 8, 8)                  │
│                                     │
│  [0] = White Pawns    [6] = Black Pawns    │
│  [1] = White Knights  [7] = Black Knights  │
│  [2] = White Bishops  [8] = Black Bishops  │
│  [3] = White Rooks    [9] = Black Rooks    │
│  [4] = White Queens   [10]= Black Queens   │
│  [5] = White King     [11]= Black King     │
│  [12]= Castling Rights                     │
│  [13]= En Passant + Metadata               │
└─────────────────────────────────────┘
```

### 2. Move Encoding

```
Chess Move: e2e4
   ↓
┌─────────────────────────────────────┐
│  UCI: "e2e4"                        │
│  From Square: e2 = 12               │
│  To Square:   e4 = 28               │
└─────────────────────────────────────┘
   ↓  encode_move()
┌─────────────────────────────────────┐
│  Policy Index: 12 * 64 + 28 = 796   │
│  UCI String: "e2e4"                 │
└─────────────────────────────────────┘
```

### 3. Value Encoding

```
Stockfish Score: +150 centipawns
   ↓
┌─────────────────────────────────────┐
│  Centipawn Score: 150 cp            │
│  (White is up 1.5 pawns)            │
└─────────────────────────────────────┘
   ↓  centipawns_to_value()
┌─────────────────────────────────────┐
│  Normalized Value: +0.15            │
│  (Range: -1.0 to +1.0)              │
└─────────────────────────────────────┘
```

## Neural Network Architecture

```
Input: Board Tensor (14, 8, 8)
   ↓
┌────────────────────────────┐
│  Conv2d(14 → 256, 3×3)     │  Initial convolution
│  BatchNorm2d               │
│  ReLU                      │
└────────────┬───────────────┘
             │
   ┌─────────┴─────────┐
   │  Residual Blocks  │  × 10
   │                   │
   │  Conv → BN → ReLU │
   │  Conv → BN        │
   │  + Skip           │
   │  ReLU             │
   └─────────┬─────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌─────────┐    ┌──────────┐
│ Policy  │    │  Value   │
│  Head   │    │   Head   │
│         │    │          │
│ Conv2d  │    │ Conv2d   │
│ Linear  │    │ Linear   │
│ (4096)  │    │ Tanh     │
│         │    │ (1)      │
└────┬────┘    └────┬─────┘
     │              │
     ▼              ▼
  Move Probs    Position Eval
  (4096 dim)    (-1 to +1)
```

## Parallel Generation Architecture

```
Main Process
     │
     ├─── Spawn Worker 1 ─── Stockfish Instance 1
     │         │
     │         └─── Generate Game 1, 5, 9, 13...
     │
     ├─── Spawn Worker 2 ─── Stockfish Instance 2
     │         │
     │         └─── Generate Game 2, 6, 10, 14...
     │
     ├─── Spawn Worker 3 ─── Stockfish Instance 3
     │         │
     │         └─── Generate Game 3, 7, 11, 15...
     │
     └─── Spawn Worker 4 ─── Stockfish Instance 4
               │
               └─── Generate Game 4, 8, 12, 16...

All results collected → Merged → Saved to NPZ
```

## File Interaction Map

```
User Interaction
     │
     ├─── quick_start.py ──────┐
     │                         │
     ├─── generate_dataset.py ─┤
     │                         │
     └─── config_template.py ──┘
                               │
                               ▼
                      DatasetConfig
                               │
                               ▼
                    generate_dataset_parallel()
                               │
                               ▼
                       Stockfish Engine
                               │
                               ▼
                        Dataset (.npz)
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
              data_loader.py      visualize_data.py
                     │
                     ▼
            ChessDataset class
                     │
                     ▼
            example_training.py
                     │
                     ▼
              Trained Model
                     │
                     ▼
            Your Bot (main.py)
```

## Configuration Flow

```
┌─────────────────────────────┐
│  Choose Configuration       │
│                             │
│  • Tiny Test (10 games)     │
│  • Small (100 games)        │
│  • Medium (1k games)        │
│  • Large (10k games)        │
│  • ChessHacks (5k games)    │
│  • Custom                   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Set Hardware Resources     │
│                             │
│  • parallel_workers         │
│  • threads_per_worker       │
│  • hash_mb                  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Set Search Budgets         │
│                             │
│  Option A: Depth-based      │
│  • play_depth               │
│  • label_depth              │
│                             │
│  Option B: Time-based       │
│  • play_time_ms             │
│  • label_time_ms            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Set Diversity Options      │
│                             │
│  • opening_variety          │
│  • opening_moves            │
│  • max_moves_per_game       │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Set Output Options         │
│                             │
│  • output_dir               │
│  • save_every               │
└────────────┬────────────────┘
             │
             ▼
       Generate Dataset!
```

## Complete System Components

### Core Engine
- `generate_dataset.py` - Main generation logic
- `data_loader.py` - Data loading utilities
- `example_training.py` - Training implementation

### Configuration
- `config_template.py` - Pre-made configs
- `DatasetConfig` - Configuration dataclass

### Utilities
- `visualize_data.py` - Data inspection
- `test_installation.py` - System verification
- `quick_start.py` - Interactive setup

### Documentation
- `README.md` - Usage guide
- `INSTALL.md` - Installation
- `PROJECT_OVERVIEW.md` - High-level design
- `ARCHITECTURE.md` - This file
- `GET_STARTED.md` - Quick start
- `DATASET_GENERATOR_SUMMARY.md` - Summary

### Setup
- `setup.bat` - Windows setup
- `setup.sh` - Unix setup
- `requirements.txt` - Dependencies

## Technology Stack

```
┌─────────────────────────────────────┐
│          Application Layer          │
│                                     │
│  Python Scripts                     │
│  • Generation                       │
│  • Training                         │
│  • Visualization                    │
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│         Framework Layer             │
│                                     │
│  • PyTorch (neural networks)        │
│  • NumPy (numerical computing)      │
│  • python-chess (chess logic)       │
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│          Engine Layer               │
│                                     │
│  • Stockfish (position evaluation)  │
│  • python-chess.engine (interface)  │
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│          System Layer               │
│                                     │
│  • Python 3.8+                      │
│  • OS (Windows/Mac/Linux)           │
│  • CPU (multi-core recommended)     │
└─────────────────────────────────────┘
```

## Scalability Model

```
Single Core:
  1 Stockfish Instance
  → 1 game at a time
  → ~1000 games/day

4 Cores:
  4 Stockfish Instances
  → 4 games in parallel
  → ~4000 games/day

16 Cores (Cloud):
  16 Stockfish Instances
  → 16 games in parallel
  → ~16000 games/day

64 Cores (Big Cloud):
  64 Stockfish Instances
  → 64 games in parallel
  → ~64000 games/day
```

## Quality vs Speed Tradeoff

```
Fast (Low Quality)
  play_depth=6, label_depth=8
  → ~1500 ELO play
  → ~1800 ELO labels
  → 2x faster

Balanced (Good Quality)
  play_depth=10, label_depth=12
  → ~2000 ELO play
  → ~2400 ELO labels
  → 1x (baseline)

Slow (High Quality)
  play_depth=14, label_depth=18
  → ~2400 ELO play
  → ~3000 ELO labels
  → 0.3x slower
```

## Memory Usage

```
Per Stockfish Instance:
  • hash_mb: 128-512 MB
  • Stack: ~10 MB
  • Total: ~150-550 MB

Per Dataset Position:
  • Board: 14×8×8×4 = 1.8 KB
  • Policy: 4 bytes
  • Value: 4 bytes
  • Metadata: ~100 bytes
  • Total: ~2 KB

For 100k positions:
  • Raw data: ~200 MB
  • Compressed NPZ: ~50-100 MB

For parallel_workers=8:
  • Stockfish: 8 × 512 MB = 4 GB
  • Python overhead: ~500 MB
  • Dataset buffer: ~500 MB
  • Total: ~5 GB
```

This architecture is designed for:
✅ **Scalability** - From laptop to cloud
✅ **Flexibility** - Many configuration options
✅ **Robustness** - Error handling and checkpointing
✅ **Performance** - Parallel generation
✅ **Quality** - Engine-level training data

