#!/usr/bin/env python3
"""
Quick start script to generate your first chess dataset!
"""

import sys
from pathlib import Path
from generate_dataset import DatasetConfig, generate_dataset_parallel


def test_stockfish(stockfish_path: str) -> bool:
    """Test if Stockfish is accessible."""
    import chess.engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.quit()
        return True
    except Exception as e:
        print(f"❌ Error: Could not access Stockfish at '{stockfish_path}'")
        print(f"   {e}")
        return False


def main():
    print("=" * 80)
    print("🚀 Chess Dataset Generator - Quick Start")
    print("=" * 80)
    
    # Get Stockfish path from user
    print("\n📍 Step 1: Locate Stockfish")
    print("\nCommon locations:")
    print("  • Windows: C:\\stockfish\\stockfish.exe")
    print("  • macOS:   /usr/local/bin/stockfish (or /opt/homebrew/bin/stockfish)")
    print("  • Linux:   /usr/bin/stockfish")
    print("\nOr just 'stockfish' if it's in your PATH")
    
    stockfish_path = input("\nEnter Stockfish path [stockfish]: ").strip()
    if not stockfish_path:
        stockfish_path = "stockfish"
    
    # Test Stockfish
    print(f"\n🔍 Testing Stockfish at: {stockfish_path}")
    if not test_stockfish(stockfish_path):
        print("\n❌ Stockfish test failed!")
        print("\nTroubleshooting:")
        print("  1. Make sure Stockfish is installed")
        print("     → Download: https://stockfishchess.org/download/")
        print("  2. Provide the full path to the executable")
        print("  3. On Windows, use forward slashes or raw strings: r'C:\\path\\to\\stockfish.exe'")
        sys.exit(1)
    
    print("✅ Stockfish is working!")
    
    # Choose dataset size
    print("\n📊 Step 2: Choose Dataset Size")
    print("\nOptions:")
    print("  1. 🧪 Tiny Test     (~10 games, ~500 positions, ~2 min)")
    print("  2. 🏃 Small        (~100 games, ~5k positions, ~15 min)")
    print("  3. 🚶 Medium       (~1k games, ~50k positions, ~2 hours)")
    print("  4. 🏋️  Large        (~10k games, ~500k positions, requires cloud)")
    
    choice = input("\nChoose [1-4] (default: 1): ").strip()
    if not choice:
        choice = "1"
    
    # Configure
    configs = {
        "1": DatasetConfig(
            stockfish_path=stockfish_path,
            threads=1,
            hash_mb=128,
            play_depth=8,
            label_depth=8,
            num_games=10,
            max_moves=50,
            opening_variety=True,
            opening_moves=4,
            output_dir="chess_dataset_tiny",
            save_every=5,
            parallel_workers=1,
        ),
        "2": DatasetConfig(
            stockfish_path=stockfish_path,
            threads=1,
            hash_mb=256,
            play_depth=10,
            label_depth=12,
            num_games=100,
            max_moves=100,
            opening_variety=True,
            opening_moves=4,
            output_dir="chess_dataset_small",
            save_every=25,
            parallel_workers=1,
        ),
        "3": DatasetConfig(
            stockfish_path=stockfish_path,
            threads=1,
            hash_mb=256,
            play_depth=10,
            label_depth=12,
            num_games=1000,
            max_moves=100,
            opening_variety=True,
            opening_moves=4,
            output_dir="chess_dataset_medium",
            save_every=100,
            parallel_workers=4,  # Parallel recommended
        ),
        "4": DatasetConfig(
            stockfish_path=stockfish_path,
            threads=1,
            hash_mb=512,
            play_depth=10,
            label_depth=14,
            num_games=10000,
            max_moves=100,
            opening_variety=True,
            opening_moves=6,
            output_dir="chess_dataset_large",
            save_every=500,
            parallel_workers=8,  # Cloud recommended
        ),
    }
    
    config = configs.get(choice, configs["1"])
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 Configuration Summary")
    print("=" * 80)
    print(f"  Stockfish:       {config.stockfish_path}")
    print(f"  Games:           {config.num_games:,}")
    print(f"  Search depth:    {config.play_depth} (play) / {config.label_depth} (label)")
    print(f"  Parallel workers: {config.parallel_workers}")
    print(f"  Output dir:      {config.output_dir}")
    print(f"  Est. positions:  ~{config.num_games * 50:,}")
    
    # Confirm
    print("\n" + "=" * 80)
    confirm = input("Start generation? [Y/n]: ").strip().lower()
    if confirm and confirm not in ['y', 'yes']:
        print("Cancelled.")
        sys.exit(0)
    
    # Generate!
    print("\n" + "=" * 80)
    print("🎮 Starting Generation...")
    print("=" * 80)
    
    try:
        generate_dataset_parallel(config)
        
        print("\n" + "=" * 80)
        print("✅ SUCCESS! Dataset generated!")
        print("=" * 80)
        print(f"\n📁 Your dataset is in: {config.output_dir}/")
        print(f"   → dataset_final.npz")
        print(f"   → stats_final.json")
        print(f"   → config.json")
        
        print("\n📚 Next steps:")
        print("   1. Visualize: python visualize_data.py {}/dataset_final.npz".format(config.output_dir))
        print("   2. Train:     python example_training.py")
        print("   3. Scale up:  Modify config and generate more data!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        print(f"   → Check {config.output_dir}/ for checkpoint files")
    except Exception as e:
        print(f"\n\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

