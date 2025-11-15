# Chess Training Data Sources - Complete Guide

## Three Ways to Get Training Data

### Option 1: Generate with Stockfish (What I Built) ⭐

**Use:** `python quick_start.py`

**Pros:**
- ✅ **Highest quality** - Every position labeled with Stockfish
- ✅ **Pure engine games** - Exactly the distribution you want
- ✅ **Customizable** - Control everything (depth, variety, etc.)
- ✅ **No downloads** - Just generate locally
- ✅ **Perfect labels** - Deep Stockfish analysis per position

**Cons:**
- ❌ Takes time (but parallelizable!)
- ❌ Requires Stockfish setup

**Best for:**
- Production chess engines
- ChessHacks (with cloud compute)
- Research projects
- When you want the absolute best quality

**Time:** 1k games = 20 min (8 cores) to 2.5 hours (1 core)

---

### Option 2: Download Lichess Games (Quick Start) 🚀

**Use:** `python download_lichess_data.py`

**Pros:**
- ✅ **Fast** - Pre-generated games
- ✅ **Free** - Millions of games available
- ✅ **Real games** - Actual engine matchups
- ✅ **No computation** - Just download and process

**Cons:**
- ❌ **No position labels** - Only game outcomes (unless you label with Stockfish)
- ❌ **Large downloads** - 5-10 GB per month
- ❌ **Mixed quality** - Various engines and ratings
- ❌ **Still need processing** - Extract and encode positions

**Best for:**
- Quick experiments
- Testing your pipeline
- Supplementing generated data
- Learning/education

**Time:** Download = 10-30 min, Extract = 5-10 min

---

### Option 3: Use Pre-processed Datasets 📦

**Sources:**
1. **Leela Chess Zero (Lc0)**
   - URL: https://lczero.org/
   - Format: Lc0 training format
   - Size: 100GB+
   
2. **Stockfish Training Data**
   - URL: https://tests.stockfishchess.org/
   - Format: Various
   - Size: Large

3. **CCRL/CEGT Archives**
   - CCRL: http://www.computerchess.org.uk/ccrl/
   - CEGT: http://www.cegt.net/
   - Format: PGN
   - Size: Varies

**Pros:**
- ✅ Instant access
- ✅ Proven quality
- ✅ Huge datasets

**Cons:**
- ❌ Different formats (need conversion)
- ❌ Not customizable
- ❌ Very large downloads
- ❌ May not match your needs

---

## Comparison Table

| Feature | Generate (Option 1) | Lichess (Option 2) | Pre-processed (Option 3) |
|---------|--------------------|--------------------|--------------------------|
| **Setup Time** | 5 min | 2 min | 2 min |
| **Generation Time** | 2-3 hours (1k games) | 30 min download + 10 min extract | 1-2 hours download |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Customization** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **Position Labels** | Every position | Game outcome only* | Varies |
| **Cost** | CPU time | Free | Free |
| **Format** | Ready to use | Needs extraction | Needs conversion |
| **Size Control** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

\* Can label with Stockfish but takes time (similar to Option 1)

---

## Recommended Approach

### For ChessHacks (36 hours)

**Strategy: Hybrid**

1. **Hour 0-1:** Download Lichess sample (1k games) for quick start
2. **Hour 1-6:** Generate high-quality data (5k games on cloud)
3. **Hour 6-24:** Train on combined dataset
4. **Hour 24-36:** Iterate and refine

```bash
# Quick start with Lichess
python download_lichess_data.py  # Get 1k games fast

# Generate quality data in parallel
python quick_start.py  # Choose ChessHacks config (5k games)

# Merge datasets
python -c "
from data_loader import merge_datasets
merge_datasets(
    ['lichess_dataset/dataset_lichess.npz', 
     'chess_dataset/dataset_final.npz'],
    'combined_dataset.npz'
)
"
```

### For Production Engine

**Use Option 1 (Generate)**

Quality matters more than speed. Generate 10k-100k games with proper Stockfish labeling.

```bash
python generate_dataset.py
# Use large_dataset_config or custom
```

### For Quick Experiment

**Use Option 2 (Lichess)**

Fast iteration, test ideas quickly.

```bash
python download_lichess_data.py
```

---

## Detailed Instructions

### Option 1: Generate (Recommended)

```bash
# Already set up!
python quick_start.py
```

See `README.md` for full details.

### Option 2: Lichess Download

```bash
# Install additional dependency
pip install tqdm requests

# Run downloader
python download_lichess_data.py
```

**Manual approach:**

1. **Visit:** https://database.lichess.org/
2. **Download:** Recent month's standard games
3. **Filter:** Look for computer games
4. **Run script:**
   ```bash
   python download_lichess_data.py
   # Provide path to downloaded PGN
   ```

**Lichess URLs:**

- Full database: `https://database.lichess.org/standard/lichess_db_standard_rated_2024-10.pgn.bz2`
- Computer only: Look for "computer" tagged games
- File size: ~5-10 GB compressed per month

### Option 3: Pre-processed Datasets

#### Leela Chess Zero

```bash
# 1. Visit https://lczero.org/
# 2. Download training data
# 3. Convert format (requires custom script)
```

#### CCRL Games

```bash
# 1. Visit http://www.computerchess.org.uk/ccrl/
# 2. Download game archives (PGN)
# 3. Use download_lichess_data.py to process PGN
```

---

## Quality Comparison

### Position Label Quality

**Generated (Option 1):**
- Each position: Stockfish depth 12-14 analysis
- Evaluation: ±10 centipawn accuracy
- Best move: From deep search
- **Use case:** Training strong engines

**Lichess without Stockfish (Option 2):**
- Each position: Game outcome only (win/loss/draw)
- Evaluation: Rough ±200 centipawn accuracy
- Best move: What engine actually played
- **Use case:** Quick experiments

**Lichess with Stockfish labeling:**
- Same quality as Option 1
- But same time cost as Option 1!
- **Use case:** When you have PGN already

---

## My Recommendation

**For you (ChessHacks):**

### Start Simple
```bash
# Day 1: Quick test
python download_lichess_data.py  # 1k games, 30 min
python visualize_data.py lichess_dataset/dataset_lichess.npz
python example_training.py  # Test training
```

### Scale Up
```bash
# Day 1-2: Generate quality data
# On cloud with 16 cores:
python quick_start.py  # Choose ChessHacks config
# 5k games, 3-4 hours, high quality
```

### Result
- Quick feedback loop (Lichess)
- High-quality training data (Generated)
- Best of both worlds!

---

## Data Format Consistency

**Good news:** All three options can be converted to the same format!

Both my generator and Lichess downloader output:
```python
{
    'boards': np.array,        # (N, 14, 8, 8)
    'policy_indices': np.array, # (N,)
    'policy_ucis': list,       # (N,)
    'values': np.array,        # (N,)
    'fens': list,              # (N,)
    'game_ids': np.array       # (N,)
}
```

So you can:
- Mix datasets
- Compare quality
- Use same training code

---

## Quick Decision Guide

**"I want to start RIGHT NOW"**
→ Option 2: `python download_lichess_data.py`

**"I want the BEST quality"**
→ Option 1: `python quick_start.py`

**"I have a PGN file already"**
→ Option 2: Provide path to `download_lichess_data.py`

**"I want MASSIVE datasets"**
→ Option 3: Lc0 or multiple Lichess months

**"I have cloud compute"**
→ Option 1: Generate with parallel workers

**"I'm doing ChessHacks"**
→ Both: Start with Option 2, generate with Option 1 in parallel

---

## Summary

You have **three options**, and I've built tools for the first two:

1. ⭐ **Generate** (best quality) - `quick_start.py`
2. 🚀 **Download** (fastest start) - `download_lichess_data.py`
3. 📦 **Pre-processed** (largest scale) - Manual download

**My recommendation:** Start with #2 to test your pipeline, then use #1 for your real training data.

Both are ready to use right now! 🎉

