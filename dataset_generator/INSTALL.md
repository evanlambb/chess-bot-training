# Installation Guide

Complete installation guide for the Chess Dataset Generator.

## Prerequisites

- **Python 3.8+** (3.9 or 3.10 recommended)
- **Stockfish Chess Engine**
- **4GB+ RAM** (more for parallel generation)

## Quick Install

### Windows

1. **Open PowerShell or Command Prompt in the `dataset_generator` folder**

2. **Run setup:**
```cmd
setup.bat
```

3. **Run quick start:**
```cmd
python quick_start.py
```

### macOS / Linux

1. **Open terminal in the `dataset_generator` folder**

2. **Make scripts executable:**
```bash
chmod +x setup.sh quick_start.py
```

3. **Run setup:**
```bash
./setup.sh
```

4. **Run quick start:**
```bash
python3 quick_start.py
```

## Manual Installation

### Step 1: Install Python Dependencies

```bash
# From the dataset_generator directory
pip install -r requirements.txt
```

**Required packages:**
- `chess` - Python chess library
- `numpy` - Numerical computing
- `torch` - PyTorch (for training)

### Step 2: Install Stockfish

#### Windows

1. **Download Stockfish:**
   - Go to: https://stockfishchess.org/download/
   - Download the Windows version
   - Extract to a folder (e.g., `C:\stockfish\`)

2. **Note the path:**
   - You'll need: `C:\stockfish\stockfish.exe` (or wherever you extracted it)

#### macOS

**Option 1: Homebrew (recommended)**
```bash
brew install stockfish
```

**Option 2: Manual**
1. Download from https://stockfishchess.org/download/
2. Extract and move to `/usr/local/bin/`

#### Linux

**Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install stockfish
```

**Fedora/RHEL:**
```bash
sudo yum install stockfish
```

**Arch:**
```bash
sudo pacman -S stockfish
```

### Step 3: Verify Installation

**Test Python packages:**
```bash
python -c "import chess; import numpy; import torch; print('All packages OK!')"
```

**Test Stockfish:**
```bash
# Windows
C:\stockfish\stockfish.exe

# macOS/Linux
stockfish
```

You should see Stockfish start. Type `quit` to exit.

## First Run

### Interactive Mode (Recommended)

```bash
python quick_start.py
```

This will guide you through:
1. Locating Stockfish
2. Choosing dataset size
3. Starting generation

### Direct Mode

```python
python generate_dataset.py
```

Or edit `generate_dataset.py` and set your Stockfish path in the `main()` function.

### Using Config Templates

```python
from config_template import balanced_config
from generate_dataset import generate_dataset_parallel

config = balanced_config()
config.stockfish_path = "C:\\stockfish\\stockfish.exe"  # Update this!
generate_dataset_parallel(config)
```

## Troubleshooting

### "Stockfish not found"

**Symptom:** `FileNotFoundError: [Errno 2] No such file or directory: 'stockfish'`

**Solutions:**
1. Provide full path to Stockfish executable
2. Add Stockfish to your PATH
3. On Windows, use raw strings: `r"C:\stockfish\stockfish.exe"`

### "Module not found"

**Symptom:** `ModuleNotFoundError: No module named 'chess'`

**Solution:**
```bash
pip install -r requirements.txt
```

### "Permission denied"

**Symptom:** On macOS/Linux when running scripts

**Solution:**
```bash
chmod +x setup.sh quick_start.py
```

### Out of Memory

**Symptom:** System freezes or crashes during generation

**Solutions:**
1. Reduce `parallel_workers` (try 1 or 2)
2. Reduce `hash_mb` (try 128)
3. Reduce `num_games` to generate in batches

### Slow Generation

**Solutions:**
1. Reduce search depth (`play_depth=8, label_depth=10`)
2. Use time-based search (`play_time_ms=10, label_time_ms=30`)
3. Increase `parallel_workers` if you have multiple cores
4. Use cloud compute with more CPUs

## Cloud Setup

### General Cloud Instance

For cloud platforms (AWS, GCP, Azure, etc.):

1. **Launch instance:**
   - Ubuntu 20.04 or 22.04
   - 16+ CPU cores recommended
   - 16GB+ RAM

2. **Install dependencies:**
```bash
# Update system
sudo apt-get update

# Install Stockfish
sudo apt-get install -y stockfish

# Install Python packages
sudo apt-get install -y python3-pip
pip3 install chess numpy torch

# Clone/upload your dataset generator
# cd to dataset_generator directory
```

3. **Configure for parallel:**
```python
config.parallel_workers = 14  # Leave 2 cores for system
config.num_games = 10000
```

4. **Run in background:**
```bash
nohup python3 generate_dataset.py > output.log 2>&1 &
```

5. **Monitor progress:**
```bash
tail -f output.log
```

### Google Colab

1. **Upload files to Colab**

2. **Install in Colab notebook:**
```python
!apt-get install -y stockfish
!pip install chess numpy
```

3. **Run generation:**
```python
from generate_dataset import DatasetConfig, generate_dataset_parallel

config = DatasetConfig(
    stockfish_path="/usr/bin/stockfish",
    num_games=1000,
    parallel_workers=2,  # Colab has limited CPU
)

generate_dataset_parallel(config)
```

## System Requirements

### Minimum (for testing)
- Python 3.8+
- 2GB RAM
- 2 CPU cores
- 1GB disk space

### Recommended (for serious datasets)
- Python 3.9+
- 8GB RAM
- 8+ CPU cores
- 10GB+ disk space

### Optimal (for production)
- Python 3.10+
- 16GB+ RAM
- 16+ CPU cores
- 50GB+ disk space
- SSD storage

## Next Steps

After installation:

1. **Generate test dataset:**
   ```bash
   python quick_start.py
   # Choose option 1 (tiny test)
   ```

2. **Visualize results:**
   ```bash
   python visualize_data.py chess_dataset_tiny/dataset_final.npz
   ```

3. **Try training:**
   ```bash
   python example_training.py
   ```

4. **Scale up:**
   - Generate larger datasets
   - Use cloud compute
   - Train better models

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify all prerequisites are installed
3. Try the test dataset first (quick_start.py option 1)
4. Check file permissions and paths
5. Look at example configs in `config_template.py`

## Version Information

Test your installation:

```bash
python -c "import chess; print(f'python-chess: {chess.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"
stockfish --version  # or path to stockfish
```

Should show:
- python-chess: 1.10.0
- numpy: 1.24.0+
- torch: 2.0.0+
- Stockfish: 15+ (newer is better)

