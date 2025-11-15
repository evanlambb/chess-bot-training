#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality.
Run this after setup to ensure everything works.
"""

import sys


def test_imports():
    """Test that all required packages are installed."""
    print("Testing Python package imports...")
    
    try:
        import chess
        print(f"  ✓ python-chess {chess.__version__}")
    except ImportError as e:
        print(f"  ✗ python-chess: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    return True


def test_stockfish(path="stockfish"):
    """Test Stockfish accessibility."""
    print(f"\nTesting Stockfish at '{path}'...")
    
    try:
        import chess.engine
        engine = chess.engine.SimpleEngine.popen_uci(path)
        info = engine.id
        print(f"  ✓ Stockfish found")
        print(f"    Name: {info.get('name', 'Unknown')}")
        print(f"    Author: {info.get('author', 'Unknown')}")
        engine.quit()
        return True
    except Exception as e:
        print(f"  ✗ Stockfish error: {e}")
        print(f"\n  Troubleshooting:")
        print(f"    1. Install Stockfish: https://stockfishchess.org/download/")
        print(f"    2. Add to PATH or provide full path")
        print(f"    3. Windows: Use r'C:\\path\\to\\stockfish.exe'")
        return False


def test_encoding():
    """Test board encoding functionality."""
    print("\nTesting board encoding...")
    
    try:
        from generate_dataset import encode_board, encode_move
        import chess
        
        # Test encoding
        board = chess.Board()
        tensor = encode_board(board)
        
        if tensor.shape != (14, 8, 8):
            print(f"  ✗ Board encoding shape error: {tensor.shape}")
            return False
        
        print(f"  ✓ Board encoding works (shape: {tensor.shape})")
        
        # Test move encoding
        move = chess.Move.from_uci("e2e4")
        policy_idx, uci = encode_move(move)
        
        if uci != "e2e4":
            print(f"  ✗ Move encoding error: {uci}")
            return False
        
        print(f"  ✓ Move encoding works")
        
        return True
    except Exception as e:
        print(f"  ✗ Encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_position():
    """Generate and label a single position."""
    print("\nTesting single position generation...")
    
    try:
        import chess
        import chess.engine
        from generate_dataset import encode_board, centipawns_to_value
        
        # Try to use Stockfish
        stockfish_path = input("  Enter Stockfish path [stockfish]: ").strip()
        if not stockfish_path:
            stockfish_path = "stockfish"
        
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        board = chess.Board()
        
        # Analyze position
        info = engine.analyse(board, chess.engine.Limit(depth=10))
        score = info.get("score")
        
        if score:
            cp_score = score.white().score(mate_score=10000)
            value = centipawns_to_value(cp_score)
            print(f"  ✓ Position analysis works")
            print(f"    Score: {cp_score} centipawns")
            print(f"    Value: {value:.3f}")
        
        # Get best move
        result = engine.play(board, chess.engine.Limit(depth=10))
        print(f"    Best move: {result.move.uci()}")
        
        engine.quit()
        
        print(f"  ✓ Full pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Position generation failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Chess Dataset Generator - Installation Test")
    print("=" * 80)
    print()
    
    results = []
    
    # Test imports
    results.append(("Package Imports", test_imports()))
    
    # Test Stockfish
    sf_path = input("\nEnter Stockfish path to test [stockfish]: ").strip()
    if not sf_path:
        sf_path = "stockfish"
    results.append(("Stockfish", test_stockfish(sf_path)))
    
    # Test encoding
    results.append(("Encoding Functions", test_encoding()))
    
    # Ask about full test
    if results[1][1]:  # If Stockfish test passed
        do_full = input("\nRun full pipeline test? (may take 10-20 seconds) [Y/n]: ").strip().lower()
        if not do_full or do_full in ['y', 'yes']:
            results.append(("Full Pipeline", test_single_position()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:<10} {name}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to generate datasets.")
        print("\nNext steps:")
        print("  1. Run: python quick_start.py")
        print("  2. Or edit generate_dataset.py and run it directly")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\nSee INSTALL.md for troubleshooting help.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

