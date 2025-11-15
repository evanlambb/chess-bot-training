"""
Utility to merge multiple chess datasets into one.
Combines multiple .npz dataset files while preserving all samples and adjusting game IDs.
"""

import numpy as np
from pathlib import Path
from typing import List
import json
import argparse


def merge_datasets(dataset_paths: List[str], output_path: str) -> None:
    """
    Merge multiple chess datasets into a single dataset.
    
    Args:
        dataset_paths: List of paths to .npz dataset files
        output_path: Path where merged dataset will be saved
    """
    print(f"Merging {len(dataset_paths)} datasets...")
    
    all_boards = []
    all_policy_indices = []
    all_policy_ucis = []
    all_values = []
    all_fens = []
    all_game_ids = []
    
    game_id_offset = 0
    
    for i, dataset_path in enumerate(dataset_paths):
        print(f"\nLoading dataset {i+1}/{len(dataset_paths)}: {dataset_path}")
        
        data = np.load(dataset_path, allow_pickle=True)
        
        # Load arrays
        boards = data['boards']
        policy_indices = data['policy_indices']
        policy_ucis = data['policy_ucis']
        values = data['values']
        fens = data['fens']
        game_ids = data['game_ids']
        
        # Adjust game IDs to be unique across datasets
        adjusted_game_ids = game_ids + game_id_offset
        game_id_offset = adjusted_game_ids.max() + 1
        
        # Append to combined lists
        all_boards.append(boards)
        all_policy_indices.append(policy_indices)
        all_policy_ucis.append(policy_ucis)
        all_values.append(values)
        all_fens.append(fens)
        all_game_ids.append(adjusted_game_ids)
        
        print(f"  Samples: {len(boards)}")
        print(f"  Games: {len(np.unique(game_ids))}")
        print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")
    
    # Combine all datasets
    print("\nCombining datasets...")
    merged_boards = np.concatenate(all_boards, axis=0)
    merged_policy_indices = np.concatenate(all_policy_indices, axis=0)
    merged_policy_ucis = np.concatenate(all_policy_ucis, axis=0)
    merged_values = np.concatenate(all_values, axis=0)
    merged_fens = np.concatenate(all_fens, axis=0)
    merged_game_ids = np.concatenate(all_game_ids, axis=0)
    
    # Save merged dataset
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"\nSaving merged dataset to: {output_path}")
    np.savez_compressed(
        output_path,
        boards=merged_boards,
        policy_indices=merged_policy_indices,
        policy_ucis=merged_policy_ucis,
        values=merged_values,
        fens=merged_fens,
        game_ids=merged_game_ids
    )
    
    # Save statistics
    stats = {
        'num_samples': int(len(merged_boards)),
        'num_games': int(len(np.unique(merged_game_ids))),
        'num_source_datasets': len(dataset_paths),
        'source_datasets': dataset_paths,
        'board_shape': list(merged_boards.shape),
        'value_mean': float(merged_values.mean()),
        'value_std': float(merged_values.std()),
        'value_min': float(merged_values.min()),
        'value_max': float(merged_values.max()),
    }
    
    stats_path = output_path_obj.parent / f"stats_merged.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)
    print(f"Total samples: {stats['num_samples']:,}")
    print(f"Total games: {stats['num_games']:,}")
    print(f"Value mean: {stats['value_mean']:.3f} ± {stats['value_std']:.3f}")
    print(f"Value range: [{stats['value_min']:.3f}, {stats['value_max']:.3f}]")
    print(f"\nOutput: {output_path}")
    print(f"Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple chess datasets into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge two datasets
  python merge_datasets.py chess_dataset/dataset_final.npz chess_dataset_v2/dataset_final.npz -o chess_dataset_merged/dataset_final.npz
  
  # Merge multiple datasets
  python merge_datasets.py dataset1.npz dataset2.npz dataset3.npz -o merged.npz
        """
    )
    
    parser.add_argument(
        'datasets',
        nargs='+',
        help='Paths to dataset .npz files to merge'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output path for merged dataset'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    for dataset_path in args.datasets:
        if not Path(dataset_path).exists():
            print(f"Error: Dataset not found: {dataset_path}")
            return
    
    if len(args.datasets) < 2:
        print("Error: Need at least 2 datasets to merge")
        return
    
    # Merge datasets
    merge_datasets(args.datasets, args.output)


if __name__ == "__main__":
    main()

