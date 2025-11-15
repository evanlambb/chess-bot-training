"""
Utility functions for loading and using the generated chess dataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional
import json


class ChessDataset(Dataset):
    """PyTorch Dataset for chess positions."""
    
    def __init__(self, npz_path: str, transform=None):
        """
        Load dataset from NPZ file.
        
        Args:
            npz_path: Path to the .npz file
            transform: Optional transform to apply to boards
        """
        data = np.load(npz_path, allow_pickle=True)
        
        self.boards = data['boards']  # Shape: (N, 14, 8, 8)
        self.policy_indices = data['policy_indices']  # Shape: (N,)
        self.values = data['values']  # Shape: (N,)
        
        # Optional metadata
        self.fens = data.get('fens', None)
        self.policy_ucis = data.get('policy_ucis', None)
        self.game_ids = data.get('game_ids', None)
        
        self.transform = transform
        
        print(f"Loaded dataset: {len(self)} samples")
        print(f"  Board shape: {self.boards.shape}")
        print(f"  Value range: [{self.values.min():.3f}, {self.values.max():.3f}]")
    
    def __len__(self) -> int:
        return len(self.boards)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            board: Tensor of shape (14, 8, 8)
            policy: Tensor with move index
            value: Tensor with position evaluation
        """
        board = self.boards[idx]
        
        if self.transform:
            board = self.transform(board)
        
        board = torch.from_numpy(board).float()
        policy = torch.tensor(self.policy_indices[idx], dtype=torch.long)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        
        return board, policy, value
    
    def get_sample_with_metadata(self, idx: int) -> dict:
        """Get sample with all metadata for debugging."""
        return {
            'board': self.boards[idx],
            'policy_idx': self.policy_indices[idx],
            'policy_uci': self.policy_ucis[idx] if self.policy_ucis is not None else None,
            'value': self.values[idx],
            'fen': self.fens[idx] if self.fens is not None else None,
            'game_id': self.game_ids[idx] if self.game_ids is not None else None,
        }


def create_data_loader(
    npz_path: str,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for the chess dataset.
    
    Args:
        npz_path: Path to the .npz file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader instance
    """
    dataset = ChessDataset(npz_path)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    return loader


def load_dataset_stats(stats_path: str) -> dict:
    """Load dataset statistics from JSON file."""
    with open(stats_path, 'r') as f:
        return json.load(f)


def merge_datasets(npz_paths: list, output_path: str) -> None:
    """
    Merge multiple NPZ datasets into one.
    
    Args:
        npz_paths: List of paths to .npz files
        output_path: Path for merged output
    """
    print(f"Merging {len(npz_paths)} datasets...")
    
    all_boards = []
    all_policy_indices = []
    all_policy_ucis = []
    all_values = []
    all_fens = []
    all_game_ids = []
    
    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        all_boards.append(data['boards'])
        all_policy_indices.append(data['policy_indices'])
        all_values.append(data['values'])
        
        if 'policy_ucis' in data:
            all_policy_ucis.extend(data['policy_ucis'])
        if 'fens' in data:
            all_fens.extend(data['fens'])
        if 'game_ids' in data:
            all_game_ids.append(data['game_ids'])
    
    # Concatenate
    merged_boards = np.concatenate(all_boards, axis=0)
    merged_policy_indices = np.concatenate(all_policy_indices, axis=0)
    merged_values = np.concatenate(all_values, axis=0)
    
    # Save merged dataset
    np.savez_compressed(
        output_path,
        boards=merged_boards,
        policy_indices=merged_policy_indices,
        policy_ucis=all_policy_ucis if all_policy_ucis else None,
        values=merged_values,
        fens=all_fens if all_fens else None,
        game_ids=np.concatenate(all_game_ids, axis=0) if all_game_ids else None,
    )
    
    print(f"Merged dataset saved to {output_path}")
    print(f"  Total samples: {len(merged_boards)}")


# Example usage
if __name__ == "__main__":
    # Load a dataset
    dataset = ChessDataset("chess_dataset/dataset_final.npz")
    
    # Print first sample with metadata
    sample = dataset.get_sample_with_metadata(0)
    print("\nFirst sample:")
    print(f"  FEN: {sample['fen']}")
    print(f"  Move (UCI): {sample['policy_uci']}")
    print(f"  Value: {sample['value']:.3f}")
    print(f"  Board shape: {sample['board'].shape}")
    
    # Create DataLoader
    loader = create_data_loader(
        "chess_dataset/dataset_final.npz",
        batch_size=64,
        shuffle=True
    )
    
    # Iterate through one batch
    boards, policies, values = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  Boards: {boards.shape}")
    print(f"  Policies: {policies.shape}")
    print(f"  Values: {values.shape}")

