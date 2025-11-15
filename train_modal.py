"""
Chess Model Training on Modal
Trains a chess neural network using Modal's GPU infrastructure.

Usage:
    # Train with default settings (30 epochs, A10G GPU):
    modal run train_modal.py
    
    # Train with custom settings:
    modal run train_modal.py --epochs 50 --batch-size 512 --gpu a100
    
    # Upload local dataset to Modal Volume first:
    modal volume put chess-datasets dataset_generator/chess_dataset/dataset_merged_6.3k.npz chess_dataset/dataset_merged_6.3k.npz
    
    # Train detached (continues after disconnect):
    modal run train_modal.py --detach
"""

# Fix encoding for Windows
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import modal
import json
from pathlib import Path
from dataclasses import dataclass, asdict

# Modal app definition
app = modal.App("chess-model-training")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "numpy>=1.24.0,<2.0.0",
        "tqdm>=4.65.0",
    )
)

# Volume for datasets and model checkpoints
volume = modal.Volume.from_name("chess-datasets", create_if_missing=True)
VOLUME_DIR = "/data"


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Dataset
    dataset_path: str = "chess_dataset/dataset_merged_6.3k.npz"  # Path in volume
    
    # Model architecture
    num_filters: int = 256
    num_res_blocks: int = 10
    
    # Training hyperparameters
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 0.001
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Loss weights
    policy_weight: float = 1.0
    value_weight: float = 1.0
    
    # Data
    train_split: float = 0.9
    num_workers: int = 4
    seed: int = 42
    
    # Checkpointing
    save_every: int = 5
    output_name: str = "chess_model"


# ====== Model Architecture ======

@app.cls(
    image=image,
    gpu="A10G",  # Default GPU, can be changed via CLI
    volumes={VOLUME_DIR: volume},
    timeout=86400,  # 24 hours
)
class ChessTrainer:
    """Modal class for training chess neural network."""
    
    @modal.enter()
    def setup(self):
        """Initialize imports for training."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        import numpy as np
        from tqdm import tqdm
        
        # Store imports for use in other methods
        self.torch = torch
        self.nn = nn
        self.F = F
        self.optim = optim
        self.Dataset = Dataset
        self.DataLoader = DataLoader
        self.np = np
        self.tqdm = tqdm
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("=" * 80)
        print("CHESS MODEL TRAINING ON MODAL")
        print("=" * 80)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("=" * 80)
    
    def _define_model_architecture(self):
        """Define model architecture classes."""
        nn = self.nn  # Get reference to nn module
        
        class ResBlock(nn.Module):
            def __init__(self, num_filters: int):
                super().__init__()
                self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(num_filters)
                self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(num_filters)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                residual = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += residual
                out = self.relu(out)
                return out
        
        class ChessNet(nn.Module):
            def __init__(self, num_filters: int, num_res_blocks: int):
                super().__init__()
                
                self.conv_input = nn.Sequential(
                    nn.Conv2d(14, num_filters, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU()
                )
                
                self.res_blocks = nn.ModuleList([
                    ResBlock(num_filters) for _ in range(num_res_blocks)
                ])
                
                self.policy_head = nn.Sequential(
                    nn.Conv2d(num_filters, 32, kernel_size=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(32 * 8 * 8, 4096),
                )
                
                self.value_head = nn.Sequential(
                    nn.Conv2d(num_filters, 32, kernel_size=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(32 * 8 * 8, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Tanh()
                )
            
            def forward(self, x):
                x = self.conv_input(x)
                for block in self.res_blocks:
                    x = block(x)
                policy = self.policy_head(x)
                value = self.value_head(x)
                return policy, value
        
        self.ResBlock = ResBlock
        self.ChessNet = ChessNet
    
    def _load_dataset(self):
        """Load dataset from Modal Volume."""
        
        class ChessDataset(self.Dataset):
            def __init__(inner_self, data_path: str):
                print(f"Loading dataset from: {data_path}")
                data = self.np.load(data_path, allow_pickle=True)
                
                inner_self.boards = self.torch.from_numpy(data['boards']).float()
                inner_self.policy_indices = self.torch.from_numpy(data['policy_indices']).long()
                inner_self.values = self.torch.from_numpy(data['values']).float()
                
                print(f"[OK] Loaded {len(inner_self.boards):,} positions")
                print(f"  Board shape: {inner_self.boards.shape}")
                print(f"  Value range: [{inner_self.values.min():.3f}, {inner_self.values.max():.3f}]")
                print(f"  Value mean: {inner_self.values.mean():.3f} +/- {inner_self.values.std():.3f}")
            
            def __len__(inner_self):
                return len(inner_self.boards)
            
            def __getitem__(inner_self, idx):
                return inner_self.boards[idx], inner_self.policy_indices[idx], inner_self.values[idx]
        
        # Load from volume
        dataset_path = Path(VOLUME_DIR) / self.config.dataset_path
        dataset = ChessDataset(str(dataset_path))
        
        # Split into train/val
        train_size = int(self.config.train_split * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = self.torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=self.torch.Generator().manual_seed(self.config.seed)
        )
        
        print(f"\nDataset split:")
        print(f"  Training: {len(self.train_dataset):,} positions")
        print(f"  Validation: {len(self.val_dataset):,} positions")
        
        # Create data loaders
        self.train_loader = self.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = self.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def _create_model(self):
        """Create and initialize model."""
        self.model = self.ChessNet(
            num_filters=self.config.num_filters,
            num_res_blocks=self.config.num_res_blocks
        ).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel architecture:")
        print(f"  Filters: {self.config.num_filters}")
        print(f"  Residual blocks: {self.config.num_res_blocks}")
        print(f"  Total parameters: {num_params:,}")
    
    def _create_optimizer(self):
        """Create optimizer and scheduler."""
        self.optimizer = self.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = self.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.min_lr
        )
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_policy_acc = 0
        
        progress_bar = self.tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (boards, policy_targets, value_targets) in enumerate(progress_bar):
            boards = boards.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device).unsqueeze(1)
            
            # Forward pass
            policy_logits, value_pred = self.model(boards)
            
            # Compute losses
            policy_loss = self.F.cross_entropy(policy_logits, policy_targets)
            value_loss = self.F.mse_loss(value_pred, value_targets)
            
            loss = self.config.policy_weight * policy_loss + self.config.value_weight * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            # Metrics
            policy_acc = (policy_logits.argmax(dim=1) == policy_targets).float().mean()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_policy_acc += policy_acc.item()
            
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'policy_acc': f"{policy_acc.item():.3f}",
                    'value_loss': f"{value_loss.item():.4f}"
                })
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'policy_acc': total_policy_acc / n_batches,
        }
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_policy_acc = 0
        
        with self.torch.no_grad():
            for boards, policy_targets, value_targets in self.tqdm(self.val_loader, desc="Validating"):
                boards = boards.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device).unsqueeze(1)
                
                policy_logits, value_pred = self.model(boards)
                
                policy_loss = self.F.cross_entropy(policy_logits, policy_targets)
                value_loss = self.F.mse_loss(value_pred, value_targets)
                loss = policy_loss + value_loss
                
                policy_acc = (policy_logits.argmax(dim=1) == policy_targets).float().mean()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_policy_acc += policy_acc.item()
        
        n_batches = len(self.val_loader)
        return {
            'loss': total_loss / n_batches,
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'policy_acc': total_policy_acc / n_batches,
        }
    
    @modal.method()
    def train(self, config_dict: dict):
        """Main training loop."""
        # Initialize config
        self.config = TrainingConfig(**config_dict)
        
        # Define model architecture
        self._define_model_architecture()
        
        # Load dataset
        self._load_dataset()
        
        # Create model
        self._create_model()
        
        # Create optimizer and scheduler
        self._create_optimizer()
        
        # Training state
        self.history = {
            'train_loss': [],
            'train_policy_loss': [],
            'train_value_loss': [],
            'train_policy_acc': [],
            'val_loss': [],
            'val_policy_loss': [],
            'val_value_loss': [],
            'val_policy_acc': [],
            'learning_rate': [],
        }
        self.best_val_loss = float('inf')
        
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Policy Acc: {train_metrics['policy_acc']:.3f}, "
                  f"Value Loss: {train_metrics['value_loss']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Policy Acc: {val_metrics['policy_acc']:.3f}, "
                  f"Value Loss: {val_metrics['value_loss']:.4f}")
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_policy_loss'].append(train_metrics['policy_loss'])
            self.history['train_value_loss'].append(train_metrics['value_loss'])
            self.history['train_policy_acc'].append(train_metrics['policy_acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_policy_loss'].append(val_metrics['policy_loss'])
            self.history['val_value_loss'].append(val_metrics['value_loss'])
            self.history['val_policy_acc'].append(val_metrics['policy_acc'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save checkpoint
            if epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch)
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_model("best")
                print(f"  [OK] New best model! (val_loss: {self.best_val_loss:.4f})")
        
        # Save final model
        self._save_model("final")
        self._save_history()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final policy accuracy: {val_metrics['policy_acc']:.3f}")
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_policy_acc': val_metrics['policy_acc'],
            'history': self.history
        }
    
    def _save_checkpoint(self, epoch: int):
        """Save checkpoint to volume."""
        output_dir = Path(VOLUME_DIR) / self.config.output_name
        output_dir.mkdir(exist_ok=True, parents=True)
        
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        self.torch.save(self.model.state_dict(), checkpoint_path)
        volume.commit()
        print(f"  [OK] Saved checkpoint: {checkpoint_path}")
    
    def _save_model(self, suffix: str):
        """Save model to volume."""
        output_dir = Path(VOLUME_DIR) / self.config.output_name
        output_dir.mkdir(exist_ok=True, parents=True)
        
        model_path = output_dir / f"{self.config.output_name}_{suffix}.pt"
        self.torch.save(self.model.state_dict(), model_path)
        volume.commit()
    
    def _save_history(self):
        """Save training history to volume."""
        output_dir = Path(VOLUME_DIR) / self.config.output_name
        output_dir.mkdir(exist_ok=True, parents=True)
        
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save config
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        volume.commit()
        print(f"[OK] Saved training history and config")


@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 256,
    num_filters: int = 256,
    num_res_blocks: int = 10,
    learning_rate: float = 0.001,
    dataset_path: str = "chess_dataset/dataset_merged_6.3k.npz",
    output_name: str = "chess_model",
):
    """
    Train chess model on Modal GPU.
    
    Args:
        epochs: Number of training epochs (default: 30)
        batch_size: Batch size (default: 256)
        num_filters: Number of filters in conv layers (default: 256)
        num_res_blocks: Number of residual blocks (default: 10)
        learning_rate: Initial learning rate (default: 0.001)
        dataset_path: Path to dataset in Modal Volume (default: chess_dataset/dataset_merged_6.3k.npz)
        output_name: Output model name (default: chess_model)
    
    Examples:
        # Default training (30 epochs, A10G GPU):
        modal run train_modal.py
        
        # Custom settings:
        modal run train_modal.py --epochs 50 --batch-size 512 --num-filters 128
        
        # Smaller model for faster training:
        modal run train_modal.py --num-filters 128 --num-res-blocks 5 --epochs 20
        
        # Run detached (continues after disconnect):
        modal run train_modal.py --detach
    """
    
    print("=" * 80)
    print("CHESS MODEL TRAINING ON MODAL")
    print("=" * 80)
    print(f"Configuration:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Model: {num_filters} filters, {num_res_blocks} blocks")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Dataset: {dataset_path}")
    print(f"   - Output: {output_name}")
    print("=" * 80)
    
    # Create config
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        num_filters=num_filters,
        num_res_blocks=num_res_blocks,
        learning_rate=learning_rate,
        dataset_path=dataset_path,
        output_name=output_name,
    )
    config_dict = asdict(config)
    
    # Create trainer and run
    trainer = ChessTrainer()
    results = trainer.train.remote(config_dict)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final policy accuracy: {results['final_policy_acc']:.3f}")
    print(f"\nModels saved to Modal Volume: chess-datasets/{output_name}/")
    print(f"\nTo download the model:")
    print(f"   modal volume get chess-datasets {output_name}/{output_name}_best.pt .")
    print(f"   modal volume get chess-datasets {output_name}/{output_name}_final.pt .")
    print(f"   modal volume get chess-datasets {output_name}/training_history.json .")
    print("=" * 80)


if __name__ == "__main__":
    pass

