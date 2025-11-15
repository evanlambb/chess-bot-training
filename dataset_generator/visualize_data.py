"""
Utility to visualize and inspect generated chess datasets.
"""

import numpy as np
import chess
import json
from pathlib import Path
from typing import Optional


def print_board_from_tensor(board_tensor: np.ndarray, title: str = "Board"):
    """
    Print a human-readable chess board from the encoded tensor.
    
    Args:
        board_tensor: Encoded board (14, 8, 8)
        title: Title to print above board
    """
    print(f"\n{title}")
    print("=" * 40)
    
    # Reconstruct piece positions
    board = [[' ' for _ in range(8)] for _ in range(8)]
    
    # Current player pieces (planes 0-5)
    piece_chars_current = ['♙', '♘', '♗', '♖', '♕', '♔']  # P, N, B, R, Q, K
    for piece_idx in range(6):
        for rank in range(8):
            for file in range(8):
                if board_tensor[piece_idx, rank, file] > 0:
                    board[rank][file] = piece_chars_current[piece_idx]
    
    # Opponent pieces (planes 6-11)
    piece_chars_opponent = ['♟', '♞', '♝', '♜', '♛', '♚']  # p, n, b, r, q, k
    for piece_idx in range(6):
        for rank in range(8):
            for file in range(8):
                if board_tensor[piece_idx + 6, rank, file] > 0:
                    board[rank][file] = piece_chars_opponent[piece_idx]
    
    # Print board (flip for display)
    print("  a b c d e f g h")
    for rank in range(7, -1, -1):
        print(f"{rank+1} ", end='')
        for file in range(8):
            print(f"{board[rank][file]} ", end='')
        print(f"{rank+1}")
    print("  a b c d e f g h")


def inspect_dataset(npz_path: str, num_samples: int = 5, show_boards: bool = True):
    """
    Inspect a generated dataset.
    
    Args:
        npz_path: Path to the .npz file
        num_samples: Number of random samples to display
        show_boards: Whether to print board visualizations
    """
    print(f"\nInspecting dataset: {npz_path}")
    print("=" * 80)
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    
    boards = data['boards']
    policy_indices = data['policy_indices']
    values = data['values']
    fens = data.get('fens', None)
    policy_ucis = data.get('policy_ucis', None)
    game_ids = data.get('game_ids', None)
    
    # Overall statistics
    print(f"\n[Dataset Statistics]")
    print(f"  Total positions: {len(boards):,}")
    print(f"  Board shape: {boards.shape}")
    print(f"  Policy indices shape: {policy_indices.shape}")
    print(f"  Values shape: {values.shape}")
    
    if game_ids is not None:
        num_games = len(np.unique(game_ids))
        avg_positions_per_game = len(boards) / num_games
        print(f"  Number of games: {num_games:,}")
        print(f"  Avg positions per game: {avg_positions_per_game:.1f}")
    
    # Value statistics
    print(f"\n[Value Distribution]")
    print(f"  Mean: {values.mean():.4f}")
    print(f"  Std:  {values.std():.4f}")
    print(f"  Min:  {values.min():.4f}")
    print(f"  Max:  {values.max():.4f}")
    print(f"  Median: {np.median(values):.4f}")
    
    # Value histogram
    print(f"\n  Value histogram:")
    hist, bins = np.histogram(values, bins=10)
    for i in range(len(hist)):
        bar = '#' * int(hist[i] / hist.max() * 50)
        print(f"    [{bins[i]:5.2f}, {bins[i+1]:5.2f}): {bar} {hist[i]:,}")
    
    # Policy statistics
    print(f"\n[Policy Distribution]")
    unique_moves, counts = np.unique(policy_indices, return_counts=True)
    print(f"  Unique moves seen: {len(unique_moves):,} (out of 4096 possible)")
    print(f"  Most common move indices:")
    top_indices = np.argsort(counts)[-5:][::-1]
    for idx in top_indices:
        move_idx = unique_moves[idx]
        count = counts[idx]
        from_sq = move_idx // 64
        to_sq = move_idx % 64
        print(f"    {chess.square_name(from_sq)}{chess.square_name(to_sq)}: {count:,} times")
    
    # Sample positions
    print(f"\n[Random Sample Positions]")
    sample_indices = np.random.choice(len(boards), min(num_samples, len(boards)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{num_samples} (Index: {idx})")
        print(f"{'='*80}")
        
        if fens is not None:
            fen = fens[idx]
            print(f"FEN: {fen}")
            
            # Reconstruct board for display
            board = chess.Board(fen)
            print(f"\nBoard (from FEN):")
            print(board)
        elif show_boards:
            print_board_from_tensor(boards[idx], "Board (from tensor)")
        
        if policy_ucis is not None:
            print(f"\n  Best move (UCI): {policy_ucis[idx]}")
        
        policy_idx = policy_indices[idx]
        from_sq = policy_idx // 64
        to_sq = policy_idx % 64
        print(f"  Best move (policy): {chess.square_name(from_sq)}{chess.square_name(to_sq)} (index {policy_idx})")
        print(f"  Value: {values[idx]:.4f}")
        
        if game_ids is not None:
            print(f"  Game ID: {game_ids[idx]}")


def compare_datasets(npz_paths: list):
    """Compare statistics across multiple datasets."""
    print("\n[Comparing Datasets]")
    print("=" * 80)
    
    stats = []
    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        stat = {
            'path': Path(path).name,
            'num_samples': len(data['boards']),
            'value_mean': data['values'].mean(),
            'value_std': data['values'].std(),
            'unique_moves': len(np.unique(data['policy_indices'])),
        }
        stats.append(stat)
    
    # Print comparison table
    print(f"\n{'Dataset':<30} {'Samples':<12} {'Value Mean':<12} {'Value Std':<12} {'Unique Moves':<12}")
    print("-" * 80)
    for stat in stats:
        print(f"{stat['path']:<30} {stat['num_samples']:<12,} {stat['value_mean']:<12.4f} {stat['value_std']:<12.4f} {stat['unique_moves']:<12,}")


def load_generation_stats(stats_json_path: str):
    """Load and display generation statistics."""
    with open(stats_json_path, 'r') as f:
        stats = json.load(f)
    
    print("\n[Generation Statistics]")
    print("=" * 80)
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_data.py <dataset.npz> [num_samples]")
        print("  python visualize_data.py compare <dataset1.npz> <dataset2.npz> ...")
        sys.exit(1)
    
    if sys.argv[1] == "compare":
        compare_datasets(sys.argv[2:])
    else:
        npz_path = sys.argv[1]
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        
        # Inspect main dataset
        inspect_dataset(npz_path, num_samples=num_samples)
        
        # Try to load stats file
        stats_path = Path(npz_path).parent / f"stats_{Path(npz_path).stem.replace('dataset_', '')}.json"
        if stats_path.exists():
            load_generation_stats(str(stats_path))

