"""
Configuration templates for different use cases.
Copy and modify these configs for your dataset generation needs.
"""

from generate_dataset import DatasetConfig


# ==============================================================================
# QUICK TEST: Verify everything works
# ==============================================================================
def quick_test_config() -> DatasetConfig:
    """
    Generate a tiny dataset to verify Stockfish and code work.
    ~5 minutes on single CPU.
    """
    return DatasetConfig(
        stockfish_path=r"C:\Users\evanl\stockfish\stockfish-windows-x86-64-avx2.exe",
        threads=1,
        hash_mb=128,
        play_depth=8,
        label_depth=8,
        num_games=10,
        max_moves=50,
        opening_variety=True,
        opening_moves=4,
        output_dir="chess_dataset_test",
        save_every=5,
        parallel_workers=1,
    )


# ==============================================================================
# SMALL DATASET: Good for initial training experiments
# ==============================================================================
def small_dataset_config() -> DatasetConfig:
    """
    ~50k positions, good for initial experiments.
    ~30 min on single CPU, ~5 min with 8 workers.
    """
    return DatasetConfig(
        stockfish_path=r"C:\Users\evanl\stockfish\stockfish-windows-x86-64-avx2.exe",
        threads=1,
        hash_mb=256,
        play_depth=10,
        label_depth=12,
        num_games=30,
        max_moves=100,
        opening_variety=True,
        opening_moves=4,
        output_dir="chess_dataset_small",
        save_every=100,
        parallel_workers=1,  # Set to 4-8 for faster generation
    )


# ==============================================================================
# MEDIUM DATASET: Serious training
# ==============================================================================
def medium_dataset_config() -> DatasetConfig:
    """
    ~500k positions, good for competitive model.
    ~5 hours on single CPU, ~40 min with 8 workers.
    """
    return DatasetConfig(
        stockfish_path=r"C:\Users\evanl\stockfish\stockfish-windows-x86-64-avx2.exe",
        threads=1,
        hash_mb=256,
        play_depth=10,
        label_depth=14,  # Deeper labeling for better quality
        num_games=10000,
        max_moves=100,
        opening_variety=True,
        opening_moves=6,
        output_dir="chess_dataset_medium",
        save_every=500,
        parallel_workers=8,  # Recommended for this size
    )


# ==============================================================================
# LARGE DATASET: Production-quality
# ==============================================================================
def large_dataset_config() -> DatasetConfig:
    """
    ~5M positions, production quality.
    Requires cloud compute with 16+ cores.
    ~6-8 hours with 16 workers.
    """
    return DatasetConfig(
        stockfish_path=r"C:\Users\evanl\stockfish\stockfish-windows-x86-64-avx2.exe",
        threads=1,
        hash_mb=512,
        play_depth=10,
        label_depth=16,  # Strong labeling
        num_games=100000,
        max_moves=100,
        opening_variety=True,
        opening_moves=8,
        output_dir="chess_dataset_large",
        save_every=1000,
        parallel_workers=16,  # Cloud compute recommended
    )


# ==============================================================================
# TIME-BASED: More consistent across hardware
# ==============================================================================
def time_based_config() -> DatasetConfig:
    """
    Uses time-based search instead of depth.
    More consistent across different hardware.
    """
    return DatasetConfig(
        stockfish_path=r"C:\Users\evanl\stockfish\stockfish-windows-x86-64-avx2.exe",
        threads=1,
        hash_mb=256,
        play_time_ms=10,    # 10ms per move during play
        label_time_ms=50,   # 50ms for high-quality labels
        num_games=1000,
        max_moves=100,
        opening_variety=True,
        opening_moves=4,
        output_dir="chess_dataset_time",
        save_every=100,
        parallel_workers=4,
    )


# ==============================================================================
# HACKATHON: 36-hour ChessHacks optimized
# ==============================================================================
def chesshacks_config() -> DatasetConfig:
    """
    Optimized for 36-hour hackathon with cloud access.
    Generates enough data quickly while maintaining quality.
    """
    return DatasetConfig(
        stockfish_path=r"C:\Users\evanl\stockfish\stockfish-windows-x86-64-avx2.exe",
        threads=1,
        hash_mb=256,
        play_depth=8,       # Faster play
        label_depth=12,     # Good labels
        num_games=5000,     # ~250k positions
        max_moves=100,
        opening_variety=True,
        opening_moves=6,
        output_dir="chess_dataset_chesshacks",
        save_every=250,
        parallel_workers=12,  # Assuming 16-core cloud machine
    )


# ==============================================================================
# BALANCED: Good quality, reasonable time
# ==============================================================================
def balanced_config() -> DatasetConfig:
    """
    Balanced between quality and generation time.
    Good default for most use cases.
    """
    return DatasetConfig(
        stockfish_path=r"C:\Users\evanl\stockfish\stockfish-windows-x86-64-avx2.exe",
        threads=1,
        hash_mb=256,
        play_depth=10,
        label_depth=12,
        num_games=300,
        max_moves=100,
        opening_variety=True,
        opening_moves=4,
        output_dir="chess_dataset",
        save_every=200,
        parallel_workers=4,
    )


# ==============================================================================
# Usage Example
# ==============================================================================
if __name__ == "__main__":
    from generate_dataset import generate_dataset_parallel
    
    # Choose a config
    config = balanced_config()
    
    # Stockfish path already configured!
    # Using: C:\Users\evanl\stockfish\stockfish-windows-x86-64-avx2.exe
    
    # Generate dataset
    print("Starting dataset generation with config:")
    print(f"  Games: {config.num_games}")
    print(f"  Parallel workers: {config.parallel_workers}")
    print(f"  Output: {config.output_dir}")
    print(f"  Search: depth {config.play_depth}/{config.label_depth}")
    
    generate_dataset_parallel(config)

