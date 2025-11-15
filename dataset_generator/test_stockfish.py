"""
Quick test to verify Stockfish is working correctly.
"""
import chess
import chess.engine

STOCKFISH_PATH = r"C:\Users\evanl\stockfish\stockfish-windows-x86-64-avx2.exe"

def test_stockfish():
    print("=" * 60)
    print("Testing Stockfish Connection")
    print("=" * 60)
    print(f"\nStockfish path: {STOCKFISH_PATH}\n")
    
    try:
        # Connect to Stockfish
        print("Connecting to Stockfish...")
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        
        # Get engine info
        print("[OK] Successfully connected!")
        print(f"  Engine: {engine.id.get('name', 'Stockfish')}")
        print(f"  Author: {engine.id.get('author', 'Unknown')}\n")
        
        # Test a simple position
        print("Testing analysis on starting position...")
        board = chess.Board()
        info = engine.analyse(board, chess.engine.Limit(depth=10))
        
        print(f"[OK] Analysis successful!")
        print(f"  Best move: {info.get('pv', [None])[0]}")
        print(f"  Score: {info.get('score')}")
        print(f"  Depth: {info.get('depth')}\n")
        
        # Clean up
        engine.quit()
        
        print("=" * 60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou're ready to generate datasets! Try:")
        print("  python config_template.py")
        print("\nOr start with a quick test:")
        print("  python generate_dataset.py")
        
        return True
        
    except FileNotFoundError:
        print(f"[ERROR] Stockfish executable not found at:")
        print(f"  {STOCKFISH_PATH}")
        print("\nPlease check the path and try again.")
        return False
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    test_stockfish()

