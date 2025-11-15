"""
Chess Dataset Generator for Modal Cloud Compute
Generates training data using Stockfish self-play with massive parallelization.

Usage:
    # Install Modal: pip install modal
    # Set up Modal auth: modal setup
    
    # Run small test (10 games):
    modal run generate_dataset_modal.py
    
    # Run large generation (10,000 games with 50 parallel workers):
    modal run generate_dataset_modal.py --num-games 10000 --parallel-workers 50
    
    # Run detached (continues even after you disconnect):
    modal run generate_dataset_modal.py --num-games 50000 --parallel-workers 100 --detach
"""

import modal
import chess
import chess.engine
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

# Modal app definition
app = modal.App("chess-dataset-generator")

# Define the container image with all dependencies
# Use Python 3.11 for better package compatibility
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("stockfish")  # Install Stockfish via apt
    .pip_install(
        "chess==1.10.0",
        "numpy>=1.24.0,<2.0.0",  # Use flexible numpy version
    )
)

# Create a persistent volume to store generated datasets
volume = modal.Volume.from_name("chess-datasets", create_if_missing=True)
VOLUME_DIR = "/data"


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    # Stockfish settings
    stockfish_path: str = "/usr/games/stockfish"  # Default apt installation path
    threads: int = 1
    hash_mb: int = 256
    
    # Search budgets
    play_depth: int = 10
    label_depth: int = 12
    play_time_ms: Optional[int] = None
    label_time_ms: Optional[int] = None
    
    # Game generation settings
    num_games: int = 1000
    max_moves: int = 100
    opening_variety: bool = True
    opening_moves: int = 10  # Increased for diversity
    
    # Stochastic play for diversity
    stochastic_play: bool = True
    stochastic_temperature: float = 0.2
    stochastic_rate: float = 0.15
    
    # Value scaling
    value_clip: int = 1000
    
    # Dataset naming
    dataset_name: str = "chess_dataset"


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode chess board as 14x8x8 tensor"""
    planes = np.zeros((14, 8, 8), dtype=np.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            piece_type = piece.piece_type - 1
            if piece.color == board.turn:
                plane_idx = piece_type
            else:
                plane_idx = piece_type + 6
            planes[plane_idx, rank, file] = 1.0
    
    # Castling rights
    if board.has_kingside_castling_rights(board.turn):
        planes[12, 0, 7] = 1.0
    if board.has_queenside_castling_rights(board.turn):
        planes[12, 0, 0] = 1.0
    if board.has_kingside_castling_rights(not board.turn):
        planes[12, 7, 7] = 1.0
    if board.has_queenside_castling_rights(not board.turn):
        planes[12, 7, 0] = 1.0
    
    # Metadata
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        planes[13, ep_rank, ep_file] = 1.0
    
    planes[13, 0, 4] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13, 1, 4] = min(board.fullmove_number / 100.0, 1.0)
    
    return planes


def encode_move(move: chess.Move) -> tuple[int, str]:
    """Encode move as (policy_index, uci_string)"""
    from_square = move.from_square
    to_square = move.to_square
    policy_idx = from_square * 64 + to_square
    return policy_idx, move.uci()


def centipawns_to_value(cp_score: int, clip: int = 1000) -> float:
    """Convert centipawn score to value in [-1, 1]"""
    clipped = np.clip(cp_score, -clip, clip)
    return clipped / clip


@app.function(
    image=image,
    timeout=3600,  # 1 hour timeout per game batch
    cpu=2,  # 2 CPUs per worker
)
def generate_single_game(config_dict: dict, game_id: int) -> List[Dict]:
    """
    Generate a single game of Stockfish self-play.
    This runs in parallel across many Modal containers.
    """
    # Reconstruct config from dict
    config = DatasetConfig(**config_dict)
    samples = []
    
    try:
        # Initialize Stockfish
        engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        engine.configure({"Threads": config.threads, "Hash": config.hash_mb})
        
        board = chess.Board()
        
        # Random opening
        if config.opening_variety and config.opening_moves > 0:
            for _ in range(config.opening_moves):
                if not board.is_game_over():
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(np.random.choice(legal_moves))
        
        move_count = 0
        
        # Play game
        while not board.is_game_over() and move_count < config.max_moves:
            # Decide whether to use stochastic selection
            use_stochastic = config.stochastic_play and np.random.random() < config.stochastic_rate
            
            if use_stochastic:
                # Get multiple move options for stochastic selection
                if config.play_time_ms:
                    analysis = engine.analyse(
                        board,
                        chess.engine.Limit(time=config.play_time_ms / 1000.0),
                        multipv=5
                    )
                else:
                    analysis = engine.analyse(
                        board,
                        chess.engine.Limit(depth=config.play_depth),
                        multipv=5
                    )
                
                moves = []
                scores = []
                for info in analysis:
                    if 'pv' in info and len(info['pv']) > 0:
                        moves.append(info['pv'][0])
                        score = info.get('score')
                        if score:
                            cp = score.white().score(mate_score=10000)
                            if board.turn == chess.BLACK:
                                cp = -cp
                            scores.append(cp)
                        else:
                            scores.append(0)
                
                if len(moves) > 0:
                    scores_array = np.array(scores, dtype=np.float32)
                    if config.stochastic_temperature > 0:
                        # Numerically stable softmax: subtract max before exp
                        scaled_scores = scores_array / (100 * config.stochastic_temperature)
                        scaled_scores = scaled_scores - scaled_scores.max()  # Prevent overflow
                        probs = np.exp(scaled_scores)
                        probs_sum = probs.sum()
                        if probs_sum > 0:  # Check for valid probabilities
                            probs = probs / probs_sum
                            best_move = np.random.choice(moves, p=probs)
                        else:
                            best_move = moves[0]  # Fallback if all probabilities are 0
                    else:
                        best_move = moves[0]
                else:
                    play_result = engine.play(
                        board,
                        chess.engine.Limit(depth=config.play_depth)
                    )
                    best_move = play_result.move
            else:
                # Use best move
                if config.play_time_ms:
                    play_result = engine.play(
                        board,
                        chess.engine.Limit(time=config.play_time_ms / 1000.0)
                    )
                else:
                    play_result = engine.play(
                        board,
                        chess.engine.Limit(depth=config.play_depth)
                    )
                best_move = play_result.move
            
            # Label position
            if config.label_time_ms:
                info = engine.analyse(
                    board,
                    chess.engine.Limit(time=config.label_time_ms / 1000.0)
                )
            else:
                info = engine.analyse(
                    board,
                    chess.engine.Limit(depth=config.label_depth)
                )
            
            # Extract evaluation
            score = info.get("score")
            if score:
                cp_score = score.white().score(mate_score=10000)
                if board.turn == chess.BLACK:
                    cp_score = -cp_score
                value = centipawns_to_value(cp_score, config.value_clip)
            else:
                value = 0.0
            
            # Encode and store
            board_tensor = encode_board(board)
            policy_idx, policy_uci = encode_move(best_move)
            
            sample = {
                'board': board_tensor,
                'policy_idx': policy_idx,
                'policy_uci': policy_uci,
                'value': value,
                'fen': board.fen(),
                'game_id': game_id
            }
            samples.append(sample)
            
            board.push(best_move)
            move_count += 1
        
        engine.quit()
        print(f"✓ Game {game_id}: {len(samples)} positions, result: {board.result()}")
        
    except Exception as e:
        print(f"✗ Game {game_id} failed: {e}")
        return []
    
    return samples


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    timeout=14400,  # 4 hours
)
def save_batch_to_volume(
    batch_samples: List[Dict],
    dataset_name: str,
    batch_id: int
) -> str:
    """Save a batch of samples to Modal Volume"""
    if not batch_samples:
        return ""
    
    # Convert to arrays
    boards = np.array([s['board'] for s in batch_samples], dtype=np.float32)
    policy_indices = np.array([s['policy_idx'] for s in batch_samples], dtype=np.int32)
    values = np.array([s['value'] for s in batch_samples], dtype=np.float32)
    fens = [s['fen'] for s in batch_samples]
    policy_ucis = [s['policy_uci'] for s in batch_samples]
    game_ids = np.array([s['game_id'] for s in batch_samples], dtype=np.int32)
    
    # Save batch to volume
    output_dir = Path(VOLUME_DIR) / dataset_name / "batches"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    batch_path = output_dir / f"batch_{batch_id:04d}.npz"
    np.savez_compressed(
        batch_path,
        boards=boards,
        policy_indices=policy_indices,
        policy_ucis=policy_ucis,
        values=values,
        fens=fens,
        game_ids=game_ids
    )
    
    volume.commit()  # Persist to volume
    
    return str(batch_path)


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    timeout=14400,  # 4 hours
    memory=8192,  # 8GB for merging
)
def merge_batches_on_volume(
    config_dict: dict,
    dataset_name: str,
    num_batches: int
) -> str:
    """Merge all batch files into final dataset on Modal Volume"""
    batch_dir = Path(VOLUME_DIR) / dataset_name / "batches"
    output_dir = Path(VOLUME_DIR) / dataset_name
    
    if not batch_dir.exists():
        return "No batches found"
    
    print(f"📦 Merging {num_batches} batches...")
    
    # Load and concatenate all batches
    all_boards = []
    all_policy_indices = []
    all_policy_ucis = []
    all_values = []
    all_fens = []
    all_game_ids = []
    
    for i in range(num_batches):
        batch_path = batch_dir / f"batch_{i:04d}.npz"
        if batch_path.exists():
            data = np.load(batch_path, allow_pickle=True)
            all_boards.append(data['boards'])
            all_policy_indices.append(data['policy_indices'])
            all_policy_ucis.extend(data['policy_ucis'])
            all_values.append(data['values'])
            all_fens.extend(data['fens'])
            all_game_ids.append(data['game_ids'])
            print(f"   ✓ Loaded batch {i}")
    
    # Concatenate arrays
    boards = np.concatenate(all_boards, axis=0)
    policy_indices = np.concatenate(all_policy_indices, axis=0)
    values = np.concatenate(all_values, axis=0)
    game_ids = np.concatenate(all_game_ids, axis=0)
    
    # Save merged dataset
    output_path = output_dir / "dataset_final.npz"
    np.savez_compressed(
        output_path,
        boards=boards,
        policy_indices=policy_indices,
        policy_ucis=all_policy_ucis,
        values=values,
        fens=all_fens,
        game_ids=game_ids
    )
    
    # Save config and stats
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    stats = {
        'num_samples': len(boards),
        'num_games': len(set(game_ids)),
        'board_shape': list(boards.shape),
        'value_mean': float(values.mean()),
        'value_std': float(values.std()),
        'value_min': float(values.min()),
        'value_max': float(values.max()),
    }
    
    with open(output_dir / "stats_final.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Clean up batch files
    print("🧹 Cleaning up batch files...")
    for batch_file in batch_dir.glob("batch_*.npz"):
        batch_file.unlink()
    batch_dir.rmdir()
    
    volume.commit()  # Persist to volume
    
    return str(output_path)


@app.local_entrypoint()
def main(
    num_games: int = 100,
    parallel_workers: int = 10,
    dataset_name: str = "chess_dataset",
    play_depth: int = 10,
    label_depth: int = 12,
    opening_moves: int = 10,
):
    """
    Generate chess dataset using Modal's parallel compute.
    
    Args:
        num_games: Number of games to generate (default 100)
        parallel_workers: Max parallel Modal containers (default 10, can go much higher)
        dataset_name: Name for the dataset
        play_depth: Stockfish depth for move generation
        label_depth: Stockfish depth for position evaluation
        opening_moves: Number of random opening moves for diversity
    
    Examples:
        # Small test run
        modal run generate_dataset_modal.py
        
        # Generate 10k games with 50 workers
        modal run generate_dataset_modal.py --num-games 10000 --parallel-workers 50
        
        # Large scale: 50k games with 100 workers (detached)
        modal run generate_dataset_modal.py --num-games 50000 --parallel-workers 100 --detach
    """
    import time
    
    print("=" * 80)
    print("🏁 CHESS DATASET GENERATION ON MODAL")
    print("=" * 80)
    print(f"📊 Configuration:")
    print(f"   • Games to generate: {num_games:,}")
    print(f"   • Max parallel workers: {parallel_workers}")
    print(f"   • Play depth: {play_depth}")
    print(f"   • Label depth: {label_depth}")
    print(f"   • Opening moves: {opening_moves}")
    print(f"   • Dataset name: {dataset_name}")
    print("=" * 80)
    
    # Create config
    config = DatasetConfig(
        num_games=num_games,
        play_depth=play_depth,
        label_depth=label_depth,
        opening_moves=opening_moves,
        dataset_name=dataset_name,
    )
    config_dict = asdict(config)
    
    # Generate all games in parallel using Modal's .map()
    print(f"\n🚀 Launching {num_games} parallel game generation tasks...")
    start_time = time.time()
    
    # Use .map() to distribute game generation across Modal containers
    # Modal automatically manages parallelization based on available resources
    batch_samples = []
    batch_size = 1000  # Save every 1000 games to avoid memory issues
    batch_id = 0
    completed = 0
    total_positions = 0
    
    for samples in generate_single_game.map(
        [config_dict] * num_games,
        range(num_games),
        order_outputs=False,  # Don't wait for order, process as completed
    ):
        batch_samples.extend(samples)
        completed += 1
        total_positions += len(samples)
        
        # Save batch when it reaches the batch size
        if len(batch_samples) >= batch_size * 100:  # Approximately 1000 games worth of positions
            print(f"💾 Saving batch {batch_id} ({len(batch_samples):,} positions)...")
            save_batch_to_volume.remote(batch_samples, dataset_name, batch_id)
            batch_samples = []  # Clear memory
            batch_id += 1
        
        if completed % 10 == 0:
            elapsed = time.time() - start_time
            rate = completed / elapsed
            eta = (num_games - completed) / rate if rate > 0 else 0
            print(
                f"📈 Progress: {completed}/{num_games} games "
                f"({total_positions:,} positions), "
                f"{rate:.1f} games/sec, ETA: {eta/60:.1f} min"
            )
    
    # Save any remaining samples in final batch
    if batch_samples:
        print(f"💾 Saving final batch {batch_id} ({len(batch_samples):,} positions)...")
        save_batch_to_volume.remote(batch_samples, dataset_name, batch_id)
        batch_id += 1
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"✅ All games completed in {elapsed/60:.1f} minutes!")
    print(f"   • Total positions: {total_positions:,}")
    print(f"   • Avg positions/game: {total_positions/num_games:.1f}")
    print(f"   • Generation rate: {num_games/elapsed:.2f} games/sec")
    print(f"   • Batches saved: {batch_id}")
    print("=" * 80)
    
    # Merge all batches into final dataset on Modal Volume
    print("\n📦 Merging batches into final dataset...")
    output_path = merge_batches_on_volume.remote(config_dict, dataset_name, batch_id)
    
    print("\n" + "=" * 80)
    print("🎉 DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"📁 Dataset saved to Modal Volume: {output_path}")
    print(f"\nTo download the dataset:")
    print(f"   modal volume get chess-datasets {dataset_name}/dataset_final.npz .")
    print(f"   modal volume get chess-datasets {dataset_name}/stats_final.json .")
    print("=" * 80)


if __name__ == "__main__":
    # This allows running with: python generate_dataset_modal.py
    # But recommended to use: modal run generate_dataset_modal.py
    pass

