#!/usr/bin/env python3
"""
Download and convert Lichess computer games to training format.
Alternative to generating data from scratch.
"""

import chess
import chess.pgn
import requests
import bz2
import numpy as np
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm
from generate_dataset import encode_board, encode_move, centipawns_to_value
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_lichess_database(year: int = 2024, month: int = 10, output_dir: str = "lichess_data") -> Path:
    """
    Download Lichess computer games database.
    
    Args:
        year: Year to download (e.g., 2024)
        month: Month to download (1-12)
        output_dir: Directory to save downloaded files
    
    Returns:
        Path to downloaded file
    """
    # Lichess database URL pattern
    base_url = "https://database.lichess.org/standard"
    
    # Format: lichess_db_standard_rated_YYYY-MM.pgn.bz2
    filename = f"lichess_db_standard_rated_{year}-{month:02d}.pgn.bz2"
    url = f"{base_url}/{filename}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    file_path = output_path / filename
    
    if file_path.exists():
        logger.info(f"File already exists: {file_path}")
        return file_path
    
    logger.info(f"Downloading from {url}")
    logger.warning("⚠️  Warning: This file is LARGE (5-10 GB). This will take a while!")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(file_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    logger.info(f"Downloaded to {file_path}")
    return file_path


def is_computer_game(game: chess.pgn.Game) -> bool:
    """Check if a game is computer vs computer."""
    headers = game.headers
    
    # Check for computer indicators
    white_title = headers.get("WhiteTitle", "")
    black_title = headers.get("BlackTitle", "")
    
    # BOT title indicates computer
    is_bot = "BOT" in white_title and "BOT" in black_title
    
    # Also check ratings - computer games usually 2000+
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
        high_rated = white_elo >= 2000 and black_elo >= 2000
    except:
        high_rated = False
    
    return is_bot or high_rated


def extract_positions_from_pgn(
    pgn_path: Path,
    max_games: int = 1000,
    min_rating: int = 2000,
    stockfish_path: Optional[str] = None,
    label_positions: bool = False
) -> list:
    """
    Extract positions from PGN file.
    
    Args:
        pgn_path: Path to PGN file (can be .bz2)
        max_games: Maximum games to process
        min_rating: Minimum average rating
        stockfish_path: Path to Stockfish (only needed if label_positions=True)
        label_positions: Whether to label with Stockfish (slow but better)
    
    Returns:
        List of training samples
    """
    samples = []
    games_processed = 0
    
    # Open file (handle bz2 compression)
    if str(pgn_path).endswith('.bz2'):
        logger.info("Decompressing BZ2 file...")
        file_handle = bz2.open(pgn_path, 'rt')
    else:
        file_handle = open(pgn_path, 'r')
    
    # Initialize Stockfish if labeling
    engine = None
    if label_positions and stockfish_path:
        import chess.engine
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    logger.info(f"Extracting positions from {pgn_path}")
    logger.info(f"Target: {max_games} games with rating >= {min_rating}")
    
    try:
        with tqdm(total=max_games, desc="Processing games") as pbar:
            while games_processed < max_games:
                game = chess.pgn.read_game(file_handle)
                if game is None:
                    break
                
                # Filter by rating
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                    avg_rating = (white_elo + black_elo) / 2
                    
                    if avg_rating < min_rating:
                        continue
                except:
                    continue
                
                # Check if computer game (optional filter)
                if not is_computer_game(game):
                    continue
                
                # Extract positions from this game
                board = game.board()
                for move in game.mainline_moves():
                    # Encode current position
                    board_tensor = encode_board(board)
                    policy_idx, policy_uci = encode_move(move)
                    
                    # Get evaluation
                    if engine and label_positions:
                        # Use Stockfish to label
                        info = engine.analyse(board, chess.engine.Limit(depth=12))
                        score = info.get("score")
                        if score:
                            cp_score = score.white().score(mate_score=10000)
                            if board.turn == chess.BLACK:
                                cp_score = -cp_score
                            value = centipawns_to_value(cp_score)
                        else:
                            value = 0.0
                    else:
                        # Use game outcome as rough label
                        result = game.headers.get("Result", "1/2-1/2")
                        if result == "1-0":
                            value = 1.0 if board.turn == chess.WHITE else -1.0
                        elif result == "0-1":
                            value = -1.0 if board.turn == chess.WHITE else 1.0
                        else:
                            value = 0.0
                    
                    sample = {
                        'board': board_tensor,
                        'policy_idx': policy_idx,
                        'policy_uci': policy_uci,
                        'value': value,
                        'fen': board.fen(),
                        'game_id': games_processed
                    }
                    samples.append(sample)
                    
                    # Make move
                    board.push(move)
                
                games_processed += 1
                pbar.update(1)
                
                if games_processed % 100 == 0:
                    logger.info(f"Processed {games_processed} games, extracted {len(samples)} positions")
    
    finally:
        file_handle.close()
        if engine:
            engine.quit()
    
    logger.info(f"Extraction complete: {len(samples)} positions from {games_processed} games")
    return samples


def save_dataset(samples: list, output_path: str):
    """Save samples to NPZ format."""
    from generate_dataset import save_dataset as save_npz
    from pathlib import Path
    
    output_dir = Path(output_path).parent
    name = Path(output_path).stem.replace("dataset_", "")
    
    save_npz(samples, output_dir, name)


def quick_download_and_extract(
    max_games: int = 1000,
    min_rating: int = 2200,
    output_dir: str = "lichess_dataset"
):
    """
    Quick function to download and extract Lichess data.
    
    This is FASTER than generating from scratch but lower quality labels.
    """
    print("=" * 80)
    print("Lichess Dataset Downloader")
    print("=" * 80)
    print()
    print("⚠️  Note: This downloads from Lichess (5-10 GB) and may take a while.")
    print("   The file will be cached for future use.")
    print()
    
    # Download (this will be slow first time)
    # Using a smaller sample - computer games only
    # For full database, use download_lichess_database()
    
    print("For this example, we'll use a sample PGN file.")
    print("You can download full databases from: https://database.lichess.org/")
    print()
    print("Alternative: Provide your own PGN file path")
    
    pgn_path = input("Enter PGN file path (or press Enter to skip): ").strip()
    
    if not pgn_path or not Path(pgn_path).exists():
        print()
        print("To use this feature:")
        print("1. Download computer games from https://database.lichess.org/")
        print("2. Or download from CCRL/CEGT")
        print("3. Run this script again with the PGN path")
        print()
        print("Alternatively, use the generator: python quick_start.py")
        return
    
    pgn_path = Path(pgn_path)
    
    # Ask about Stockfish labeling
    use_sf = input("Use Stockfish to label positions? (better but slower) [y/N]: ").strip().lower()
    
    stockfish_path = None
    if use_sf in ['y', 'yes']:
        stockfish_path = input("Stockfish path [stockfish]: ").strip() or "stockfish"
    
    # Extract
    samples = extract_positions_from_pgn(
        pgn_path,
        max_games=max_games,
        min_rating=min_rating,
        stockfish_path=stockfish_path,
        label_positions=bool(stockfish_path)
    )
    
    # Save
    output_path = Path(output_dir) / "dataset_lichess.npz"
    save_dataset(samples, str(output_path))
    
    print()
    print("=" * 80)
    print("✅ Dataset created!")
    print("=" * 80)
    print(f"Saved to: {output_path}")
    print(f"Positions: {len(samples):,}")
    print()
    print("Next steps:")
    print(f"  python visualize_data.py {output_path}")
    print(f"  python example_training.py")


if __name__ == "__main__":
    # Example usage
    quick_download_and_extract(
        max_games=1000,
        min_rating=2200,
        output_dir="lichess_dataset"
    )

