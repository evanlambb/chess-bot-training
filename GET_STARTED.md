# Chess Dataset Generator - Getting Started

## 🚀 Quick Start

You now have a complete chess dataset generation system! Here's how to use it:

### Step 1: Install Dependencies

**Windows:**
```cmd
cd dataset_generator
setup.bat
```

**macOS/Linux:**
```bash
cd dataset_generator
chmod +x setup.sh quick_start.py
./setup.sh
```

### Step 2: Generate Your First Dataset

Run the interactive wizard:

```bash
python quick_start.py
```

Or directly edit and run:

```bash
python generate_dataset.py
```

### Step 3: Visualize the Data

```bash
python visualize_data.py chess_dataset_tiny/dataset_final.npz
```

### Step 4: Train a Model

```bash
python example_training.py
```

## 📁 What You Have

```
dataset_generator/
├── generate_dataset.py      ⭐ Main generator
├── quick_start.py           ⭐ Interactive wizard
├── data_loader.py           📦 PyTorch utilities
├── example_training.py      🎓 Training example
├── config_template.py       ⚙️  Config presets
├── visualize_data.py        📊 Inspect data
├── requirements.txt         📋 Dependencies
├── README.md                📚 Full documentation
├── INSTALL.md               🔧 Installation guide
├── PROJECT_OVERVIEW.md      📖 Architecture overview
├── setup.bat               🪟 Windows setup
└── setup.sh                🐧 Linux/Mac setup
```

## 🎯 Common Use Cases

### For ChessHacks (36-hour hackathon)

```python
# Edit config_template.py or generate_dataset.py
config = DatasetConfig(
    stockfish_path="stockfish",  # Update this!
    play_depth=8,
    label_depth=12,
    num_games=5000,
    parallel_workers=12,  # Use cloud compute
    output_dir="chess_dataset_chesshacks"
)
```

**Timeline:**
- Hour 0-2: Setup + test
- Hour 2-6: Generate on cloud (5k games)
- Hour 6-24: Train model
- Hour 24-36: Test and iterate

### For Learning/Experimentation

```bash
# Use quick_start.py and select "Tiny Test"
python quick_start.py
# Choose option 1
```

Generates ~500 positions in 2 minutes.

### For Production

```python
# Use large_dataset_config
from config_template import large_dataset_config
from generate_dataset import generate_dataset_parallel

config = large_dataset_config()
config.stockfish_path = "/path/to/stockfish"
generate_dataset_parallel(config)
```

## ⚠️ Important: Update Stockfish Path

Before running, update the Stockfish path in your chosen script:

**Windows:**
```python
config.stockfish_path = r"C:\stockfish\stockfish.exe"
```

**macOS/Linux:**
```python
config.stockfish_path = "/usr/local/bin/stockfish"
# or just "stockfish" if in PATH
```

## 📖 Documentation

- **README.md** - Complete usage guide
- **INSTALL.md** - Installation troubleshooting
- **PROJECT_OVERVIEW.md** - Architecture and design

## 🆘 Need Help?

1. **Stockfish not found?** → See INSTALL.md
2. **Out of memory?** → Reduce `parallel_workers` to 1
3. **Too slow?** → Increase `parallel_workers` or use cloud
4. **Want better data?** → Increase `label_depth` to 14-16

## 🌟 What Makes This Special?

✅ **Engine-vs-engine focused** - Not human games  
✅ **Parallel generation** - Use all your CPU cores  
✅ **Production ready** - Robust error handling  
✅ **Flexible** - Many configuration options  
✅ **Complete pipeline** - Generation → Training → Evaluation  
✅ **Well documented** - Extensive guides  

## 🎮 Next Steps

1. ✅ Generate a test dataset (100 games)
2. ✅ Visualize the data
3. ✅ Try the example training
4. ✅ Scale up to larger datasets
5. ✅ Integrate with your chess bot
6. ✅ Generate data from your model vs Stockfish
7. ✅ Iterate and improve!

## 💡 Pro Tips

- **Start small**: Test with 10-100 games first
- **Use cloud**: 16+ cores makes generation 10x faster
- **Monitor progress**: Logs show games/sec and ETA
- **Save checkpoints**: Don't lose progress if something crashes
- **Experiment**: Try different depths and configurations

## 🔗 Integration Example

To use the generated data with your bot in `my-chesshacks-bot/`:

1. **Generate dataset** (you're here!)
2. **Train model** using `example_training.py`
3. **Export model** to ONNX or save PyTorch weights
4. **Load in bot**:
   ```python
   # In my-chesshacks-bot/src/main.py
   import torch
   from dataset_generator.generate_dataset import encode_board
   
   # Load model once
   model = torch.load('chess_model_final.pt')
   model.eval()
   
   @chess_manager.entrypoint
   def test_func(ctx: GameContext):
       # Encode current position
       board_tensor = encode_board(ctx.board)
       board_tensor = torch.from_numpy(board_tensor).unsqueeze(0)
       
       # Get predictions
       with torch.no_grad():
           policy_logits, value = model(board_tensor)
       
       # Convert policy to move probabilities
       legal_moves = list(ctx.board.legal_moves)
       # ... pick best legal move from policy
       
       return best_move
   ```

Ready to build a powerful chess engine! 🚀♟️

---

**Have questions?** Check the documentation files or experiment with the code!

