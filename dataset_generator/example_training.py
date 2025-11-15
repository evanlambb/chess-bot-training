"""
Example training script using the generated dataset.
Shows how to build a simple neural network and train it.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from data_loader import create_data_loader
import time


class SimpleChessNet(nn.Module):
    """
    Simple CNN for chess position evaluation.
    
    Architecture:
    - Multiple convolutional layers
    - Separate heads for policy (move prediction) and value (position evaluation)
    """
    
    def __init__(self, num_filters: int = 256, num_res_blocks: int = 10):
        super().__init__()
        
        # Input: (batch, 14, 8, 8)
        self.conv_input = nn.Sequential(
            nn.Conv2d(14, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head (predicts best move)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096),  # 64 * 64 possible moves
        )
        
        # Value head (evaluates position)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Board tensor (batch, 14, 8, 8)
        
        Returns:
            policy_logits: Move logits (batch, 4096)
            value: Position evaluation (batch, 1)
        """
        x = self.conv_input(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value


class ResBlock(nn.Module):
    """Residual block with skip connection."""
    
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


def train_epoch(model, loader, optimizer, device, policy_weight=1.0, value_weight=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    for batch_idx, (boards, policy_targets, value_targets) in enumerate(loader):
        boards = boards.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Forward pass
        policy_logits, value_pred = model(boards)
        
        # Compute losses
        policy_loss = criterion_policy(policy_logits, policy_targets)
        value_loss = criterion_value(value_pred, value_targets)
        
        # Combined loss
        loss = policy_weight * policy_loss + value_weight * value_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: "
                  f"Loss={loss.item():.4f}, "
                  f"Policy={policy_loss.item():.4f}, "
                  f"Value={value_loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    avg_policy_loss = total_policy_loss / len(loader)
    avg_value_loss = total_value_loss / len(loader)
    
    return avg_loss, avg_policy_loss, avg_value_loss


def validate(model, loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    with torch.no_grad():
        for boards, policy_targets, value_targets in loader:
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device).unsqueeze(1)
            
            policy_logits, value_pred = model(boards)
            
            policy_loss = criterion_policy(policy_logits, policy_targets)
            value_loss = criterion_value(value_pred, value_targets)
            loss = policy_loss + value_loss
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    
    avg_loss = total_loss / len(loader)
    avg_policy_loss = total_policy_loss / len(loader)
    avg_value_loss = total_value_loss / len(loader)
    
    return avg_loss, avg_policy_loss, avg_value_loss


def main():
    """Example training loop."""
    
    # Configuration
    DATASET_PATH = "chess_dataset/dataset_final.npz"
    BATCH_SIZE = 256
    EPOCHS = 10
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.9
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    from data_loader import ChessDataset
    full_dataset = ChessDataset(DATASET_PATH)
    
    # Split into train/validation
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = SimpleChessNet(num_filters=128, num_res_blocks=5)
    model = model.to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_policy, train_value = train_epoch(
            model, train_loader, optimizer, device
        )
        
        val_loss, val_policy, val_value = validate(model, val_loader, device)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} (Policy: {train_policy:.4f}, Value: {train_value:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (Policy: {val_policy:.4f}, Value: {val_value:.4f})")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f"checkpoint_epoch_{epoch}.pt")
            print(f"  Saved checkpoint: checkpoint_epoch_{epoch}.pt")
    
    # Save final model
    torch.save(model.state_dict(), "chess_model_final.pt")
    print("\nTraining complete! Saved model to chess_model_final.pt")


if __name__ == "__main__":
    main()

