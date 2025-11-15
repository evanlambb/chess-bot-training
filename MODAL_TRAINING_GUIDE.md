# Modal Training Guide for Chess Bot

Train your chess model using Modal's cloud GPU infrastructure.

## 🚀 Quick Start (3 steps)

### Step 1: Upload Dataset to Modal Volume

```bash
cd dataset_generator/chess_dataset
modal volume put chess-datasets dataset_merged_6.3k.npz chess_dataset/dataset_merged_6.3k.npz
```

This uploads your 594K position dataset to Modal's persistent storage.

### Step 2: Run Training

```bash
# Default training (30 epochs, A10G GPU):
modal run train_modal.py

# Or run detached (continues even if you disconnect):
modal run train_modal.py --detach
```

### Step 3: Download Trained Model

After training completes (~2-3 hours):

```bash
# Download the best model
modal volume get chess-datasets chess_model/chess_model_best.pt .

# Download training history
modal volume get chess-datasets chess_model/training_history.json .
```

---

## ⚙️ Configuration Options

### Quick Examples

**Fast training** (testing, ~1 hour):
```bash
modal run train_modal.py \
  --epochs 10 \
  --num-filters 128 \
  --num-res-blocks 5 \
  --batch-size 512
```

**Full training** (best quality, ~3 hours):
```bash
modal run train_modal.py \
  --epochs 50 \
  --num-filters 256 \
  --num-res-blocks 15 \
  --batch-size 256
```

**Balanced** (recommended, ~2 hours):
```bash
modal run train_modal.py \
  --epochs 30 \
  --num-filters 256 \
  --num-res-blocks 10
```

### All Parameters

```bash
modal run train_modal.py \
  --epochs 30                  # Number of epochs (default: 30)
  --batch-size 256            # Batch size (default: 256)
  --num-filters 256           # Conv filters (default: 256)
  --num-res-blocks 10         # Residual blocks (default: 10)
  --learning-rate 0.001       # Initial LR (default: 0.001)
  --dataset-path "..."        # Dataset path in volume
  --output-name "chess_model" # Output model name
```

---

## 🎛️ GPU Options

Modal supports different GPU types. Edit `train_modal.py` line 46:

```python
# Fast and cheap (default):
gpu="A10G"  # ~$1-2 per hour, 24GB VRAM, great for this task

# Faster (overkill for this):
gpu="A100"  # ~$3-4 per hour, 40GB VRAM

# Cheaper (slower):
gpu="T4"    # ~$0.50 per hour, 16GB VRAM
```

**Recommendation**: Stick with **A10G** (default) - perfect balance of speed and cost.

---

## 📊 Expected Results

With 594K positions on A10G GPU:

| Setting | Time | Policy Acc | Cost (est.) |
|---------|------|------------|-------------|
| Fast (10 epochs, 128 filters) | ~1 hour | 40-45% | ~$1 |
| Balanced (30 epochs, 256 filters) | ~2-3 hours | 45-55% | ~$3-5 |
| Full (50 epochs, 256 filters) | ~4-5 hours | 50-60% | ~$5-8 |

---

## 📁 File Structure in Modal Volume

After training:

```
/chess-datasets/
├── chess_dataset/
│   └── dataset_merged_6.3k.npz          # Your dataset
└── chess_model/
    ├── chess_model_best.pt              # Best model (use this!)
    ├── chess_model_final.pt             # Final model
    ├── checkpoint_epoch_5.pt            # Checkpoint at epoch 5
    ├── checkpoint_epoch_10.pt           # Checkpoint at epoch 10
    ├── ...                              # More checkpoints
    ├── training_history.json            # Metrics over time
    └── training_config.json             # Config used
```

---

## 🔍 Monitoring Training

Modal will show real-time progress:

```
Epoch 1/30
100%|████████| 2344/2344 [01:45<00:00, loss=2.3456, policy_acc=0.432]

Epoch 1 Results:
  Train - Loss: 2.3456, Policy Acc: 0.432, Value Loss: 0.089
  Val   - Loss: 2.2134, Policy Acc: 0.445, Value Loss: 0.082
  ✓ New best model! (val_loss: 2.2134)
```

### View in Modal Dashboard

1. Go to: https://modal.com/apps
2. Click on "chess-model-training"
3. See live logs and GPU metrics

---

## 💰 Cost Estimation

Modal pricing (as of 2024):
- A10G: ~$1.10/hour
- Free tier: $30/month credit

Your training costs:
- **30 epochs**: ~2.5 hours = **~$2.75**
- **50 epochs**: ~4 hours = **~$4.40**

Much cheaper than Kaggle if you need > 30 hours/week!

---

## 🔧 Troubleshooting

### "Dataset not found in volume"

Check what's in your volume:
```bash
modal volume ls chess-datasets
```

Upload dataset:
```bash
modal volume put chess-datasets dataset_generator/chess_dataset/dataset_merged_6.3k.npz chess_dataset/dataset_merged_6.3k.npz
```

### "Out of memory"

Reduce batch size:
```bash
modal run train_modal.py --batch-size 128
```

Or reduce model size:
```bash
modal run train_modal.py --num-filters 128 --num-res-blocks 5
```

### "Training taking too long"

- Increase batch size: `--batch-size 512`
- Use smaller model: `--num-filters 128`
- Fewer epochs: `--epochs 20`

### Check GPU utilization

Add to `train_modal.py` in train loop:
```python
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

---

## 📈 View Training Progress

Download and plot training curves:

```bash
modal volume get chess-datasets chess_model/training_history.json .
```

Then locally:

```python
import json
import matplotlib.pyplot as plt

with open('training_history.json', 'r') as f:
    history = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train')
axes[0, 0].plot(history['val_loss'], label='Val')
axes[0, 0].set_title('Total Loss')
axes[0, 0].legend()

# Policy Accuracy
axes[0, 1].plot(history['train_policy_acc'], label='Train')
axes[0, 1].plot(history['val_policy_acc'], label='Val')
axes[0, 1].set_title('Policy Accuracy')
axes[0, 1].legend()

# Value Loss
axes[1, 0].plot(history['train_value_loss'], label='Train')
axes[1, 0].plot(history['val_value_loss'], label='Val')
axes[1, 0].set_title('Value Loss')
axes[1, 0].legend()

# Learning Rate
axes[1, 1].plot(history['learning_rate'])
axes[1, 1].set_title('Learning Rate')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
```

---

## 🎯 After Training

1. **Download model**:
   ```bash
   modal volume get chess-datasets chess_model/chess_model_best.pt dataset_generator/
   ```

2. **Update your bot** (`src/main.py` line 156):
   ```python
   MODEL_PATH = Path(__file__).parent.parent / "dataset_generator" / "chess_model_best.pt"
   ```

3. **Update model config** (`src/main.py` line 159) to match training:
   ```python
   model = SimpleChessNet(num_filters=256, num_res_blocks=10)  # Match your training config
   ```

4. **Test your bot**!

---

## 💡 Pro Tips

1. **Run detached for long training**:
   ```bash
   modal run train_modal.py --detach
   ```
   You can close your terminal and it keeps running!

2. **Save checkpoints**: Automatically saved every 5 epochs. If training crashes, you can resume.

3. **Monitor costs**: Check usage at https://modal.com/settings/usage

4. **Experiment with hyperparameters**: Try different configs to find what works best.

5. **Use the best model**: `chess_model_best.pt` usually performs better than `final`.

---

## 🔄 Resume Training (if needed)

To resume from a checkpoint (requires modifying script):

1. Download checkpoint:
   ```bash
   modal volume get chess-datasets chess_model/checkpoint_epoch_15.pt .
   ```

2. Modify `train_modal.py` to load checkpoint in `_create_model()`:
   ```python
   checkpoint = torch.load('/data/chess_model/checkpoint_epoch_15.pt')
   self.model.load_state_dict(checkpoint)
   ```

3. Adjust starting epoch and run again.

---

## 📞 Need Help?

Common issues:
- **Volume errors**: Make sure you uploaded dataset first
- **GPU not working**: Check Modal logs for CUDA errors
- **Slow training**: Try increasing batch size or using smaller model
- **Out of credits**: Add payment method at https://modal.com/settings/billing

**Modal Docs**: https://modal.com/docs

---

**Ready to train?** 🚀

```bash
# Upload dataset
modal volume put chess-datasets dataset_generator/chess_dataset/dataset_merged_6.3k.npz chess_dataset/dataset_merged_6.3k.npz

# Start training
modal run train_modal.py --detach

# Come back in 2-3 hours and download your model!
```

