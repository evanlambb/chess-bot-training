# Chess Dataset Generation with Modal

This guide explains how to use Modal for massively parallel chess dataset generation.

## What is Modal?

Modal is a serverless cloud compute platform that makes it trivial to scale Python code. For this use case:
- ✅ **Massive parallelization**: Run 50-100+ games simultaneously
- ✅ **Zero infrastructure**: No servers to manage
- ✅ **Pay-per-use**: Only pay for compute time used
- ✅ **Simple deployment**: Just add decorators to Python functions
- ✅ **Automatic scaling**: Modal handles all container orchestration

## Setup (One-time)

### 1. Install Modal
```bash
pip install modal
```

### 2. Authenticate
```bash
modal setup
```
This opens a browser to link your Modal account (free tier available).

### 3. Verify Installation
```bash
modal --help
```

## Generating Datasets

### Quick Test (10 games, ~2 minutes)
```bash
modal run dataset_generator/generate_dataset_modal.py
```

### Small Dataset (1,000 games, ~10-15 minutes with 20 workers)
```bash
modal run dataset_generator/generate_dataset_modal.py \
    --num-games 1000 \
    --parallel-workers 20 \
    --dataset-name chess_1k
```

### Medium Dataset (10,000 games, ~20-30 minutes with 50 workers)
```bash
modal run dataset_generator/generate_dataset_modal.py \
    --num-games 10000 \
    --parallel-workers 50 \
    --dataset-name chess_10k
```

### Large Dataset (50,000 games, ~1-2 hours with 100 workers)
```bash
modal run dataset_generator/generate_dataset_modal.py \
    --num-games 50000 \
    --parallel-workers 100 \
    --dataset-name chess_50k \
    --detach
```

The `--detach` flag lets the job continue even after you close your terminal!

### Custom Configuration
```bash
modal run dataset_generator/generate_dataset_modal.py \
    --num-games 20000 \
    --parallel-workers 80 \
    --dataset-name my_custom_dataset \
    --play-depth 12 \
    --label-depth 14 \
    --opening-moves 12
```

## Downloading Your Dataset

After generation completes, download from Modal's persistent volume:

```bash
# Download the main dataset file
modal volume get chess-datasets chess_10k/dataset_final.npz ./

# Download statistics
modal volume get chess-datasets chess_10k/stats_final.json ./

# Download config
modal volume get chess-datasets chess_10k/config.json ./
```

Or download the entire directory:
```bash
modal volume get chess-datasets chess_10k/ ./chess_10k/
```

## Merging with Existing Data

To combine your Modal-generated dataset with your existing 300-game dataset:

```bash
python dataset_generator/merge_datasets.py \
    chess_dataset/dataset_final.npz \
    chess_10k/dataset_final.npz \
    -o chess_dataset_merged/dataset_final.npz
```

## Cost Estimation

Modal pricing (approximate, check modal.com/pricing for latest):
- **Free tier**: $30/month credits (good for ~10k games)
- **Pay-as-you-go**: ~$0.0001-0.0003 per CPU-second
- **Estimated costs**:
  - 1,000 games: ~$1-2
  - 10,000 games: ~$10-20
  - 50,000 games: ~$50-100

*Note: Actual costs depend on game length and search depth*

## Performance Comparison

| Method | Workers | 10k Games Time | Notes |
|--------|---------|----------------|-------|
| Local Sequential | 1 | ~30-50 hours | Single machine |
| Local Parallel | 8 | ~4-6 hours | Multi-core |
| Modal | 50 | ~20-30 min | Cloud parallel |
| Modal | 100 | ~10-15 min | Max parallelization |

## Monitoring Your Job

### View logs in real-time
```bash
modal app logs chess-dataset-generator
```

### Check running functions
```bash
modal app list
```

### Stop a running job
```bash
modal app stop chess-dataset-generator
```

## Advanced: Batch Multiple Datasets

Generate multiple datasets with different configurations in parallel:

```bash
# Terminal 1: Generate dataset with depth 10
modal run dataset_generator/generate_dataset_modal.py \
    --num-games 10000 --play-depth 10 --dataset-name chess_depth10 --detach

# Terminal 2: Generate dataset with depth 12
modal run dataset_generator/generate_dataset_modal.py \
    --num-games 10000 --play-depth 12 --dataset-name chess_depth12 --detach

# Terminal 3: Generate dataset with more opening variety
modal run dataset_generator/generate_dataset_modal.py \
    --num-games 10000 --opening-moves 15 --dataset-name chess_varied --detach
```

All three will run independently in the cloud!

## Troubleshooting

### "No such command 'run'"
- Make sure Modal is installed: `pip install modal`
- Verify version: `modal --version`

### "Authentication required"
- Run: `modal setup`

### "Timeout errors"
- Increase timeout in the `@app.function()` decorator
- Reduce search depth for faster games

### "Out of memory"
- Reduce `hash_mb` in DatasetConfig
- The default 256MB is usually fine

## Tips for Best Results

1. **Start small**: Test with 10-100 games first
2. **Use detach**: For large jobs (>5k games), use `--detach`
3. **Monitor costs**: Check Modal dashboard regularly
4. **Diversity is key**: Keep `opening_moves` at 10-12
5. **Merge datasets**: Combine multiple runs for even more diversity

## Next Steps

After generating your dataset:

1. **Verify quality**: Check `stats_final.json` for value distribution
2. **Merge with old data**: Use `merge_datasets.py`
3. **Train your model**: Point training script to new dataset
4. **Generate more**: As you need, generate additional batches

Happy training! 🚀♟️

