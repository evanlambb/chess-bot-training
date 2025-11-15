"""
Chess Dataset Generator for Engine Training
Generates training data using Stockfish self-play with optional separate labeling budget.

SCALING RECOMMENDATIONS:
- For basic learning: 10,000+ games
- For decent performance: 50,000+ games  
- For strong play: 100,000+ games
- opening_moves=10-12 provides good position diversity
- stochastic_play=True prevents games from converging to same lines
- Use parallel_workers > 1 on cloud/multi-core systems for faster generation
"""

import chess
import chess.engine
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    # Stockfish settings
    stockfish_path: str = "stockfish"  # Path to stockfish executable
    threads: int = 1  # Threads per stockfish instance
    hash_mb: int = 256  # Hash table size in MB
    
    # Search budgets
    play_depth: int = 10  # Depth for generating moves during self-play
    label_depth: int = 12  # Depth for labeling positions (set same as play_depth for v1)
    play_time_ms: Optional[int] = None  # Alternative: time-based search (e.g., 20)
    label_time_ms: Optional[int] = None  # Alternative: time-based labeling (e.g., 50)
    
    # Game generation settings
    num_games: int = 1000  # Number of games to generate
    max_moves: int = 100  # Maximum moves per game (200 plies)
    opening_variety: bool = True  # Use random opening book for variety
    opening_moves: int = 10  # Number of random opening moves (8-12 recommended for diversity)
    
    # Stochastic play for diversity (prevents converging to same lines)
    stochastic_play: bool = True  # Enable probabilistic move selection
    stochastic_temperature: float = 0.2  # Temperature for move selection (0=greedy, higher=more random)
    stochastic_rate: float = 0.15  # Probability of using stochastic selection (vs best move)
    
    # Dataset settings
    output_dir: str = "chess_dataset"
    save_every: int = 100  # Save checkpoint every N games
    
    # Parallel processing
    parallel_workers: int = 1  # Number of parallel game generators (use > 1 for cloud)
    
    # Value scaling
    value_clip: int = 1000  # Clip centipawn scores to this range
    

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode chess board as neural network input tensor.
    
    Returns 14 planes of 8x8:
    - Planes 0-5: Current player's pieces (P, N, B, R, Q, K)
    - Planes 6-11: Opponent's pieces (P, N, B, R, Q, K)
    - Plane 12: Castling rights (4 values in corner squares)
    - Plane 13: En passant square (if any), turn color, move count
    
    Shape: (14, 8, 8)
    """
    planes = np.zeros((14, 8, 8), dtype=np.float32)
    
    # Get piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            # Determine plane index
            piece_type = piece.piece_type - 1  # 0-5 for P,N,B,R,Q,K
            if piece.color == board.turn:
                plane_idx = piece_type  # Current player: planes 0-5
            else:
                plane_idx = piece_type + 6  # Opponent: planes 6-11
            
            planes[plane_idx, rank, file] = 1.0
    
    # Castling rights plane (plane 12)
    if board.has_kingside_castling_rights(board.turn):
        planes[12, 0, 7] = 1.0  # Kingside
    if board.has_queenside_castling_rights(board.turn):
        planes[12, 0, 0] = 1.0  # Queenside
    if board.has_kingside_castling_rights(not board.turn):
        planes[12, 7, 7] = 1.0
    if board.has_queenside_castling_rights(not board.turn):
        planes[12, 7, 0] = 1.0
    
    # Metadata plane (plane 13)
    # En passant
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        planes[13, ep_rank, ep_file] = 1.0
    
    # Side to move (fill a specific location)
    planes[13, 0, 4] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Move count (normalized, in another specific location)
    planes[13, 1, 4] = min(board.fullmove_number / 100.0, 1.0)
    
    return planes


def encode_move(move: chess.Move) -> Tuple[int, str]:
    """
    Encode a move in two formats:
    1. Policy index: from_square * 64 + to_square (0-4095)
    2. UCI string: e.g., "e2e4", "e7e8q"
    
    Returns: (policy_index, uci_string)
    """
    from_square = move.from_square
    to_square = move.to_square
    policy_idx = from_square * 64 + to_square
    uci = move.uci()
    
    return policy_idx, uci


def centipawns_to_value(cp_score: int, clip: int = 1000) -> float:
    """
    Convert centipawn score to value in [-1, 1].
    Clips extreme values and applies tanh scaling.
    """
    clipped = np.clip(cp_score, -clip, clip)
    # Simple linear scaling
    return clipped / clip


def generate_single_game(
    config: DatasetConfig,
    game_id: int,
    stockfish_path: str
) -> List[Dict]:
    """
    Generate a single game of Stockfish self-play and extract training samples.
    
    Returns list of training samples, each containing:
    - board_tensor: encoded board state
    - policy_target: move index
    - policy_uci: move in UCI format
    - value_target: position evaluation
    - fen: FEN string for debugging
    """
    samples = []
    
    try:
        # Initialize Stockfish
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": config.threads, "Hash": config.hash_mb})
        
        # Start new game
        board = chess.Board()
        
        # Optional: random opening for variety
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
                # Get multiple move options with scores for stochastic selection
                if config.play_time_ms:
                    analysis = engine.analyse(
                        board,
                        chess.engine.Limit(time=config.play_time_ms / 1000.0),
                        multipv=5  # Get top 5 moves
                    )
                else:
                    analysis = engine.analyse(
                        board,
                        chess.engine.Limit(depth=config.play_depth),
                        multipv=5  # Get top 5 moves
                    )
                
                # Extract moves and scores
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
                
                # Apply temperature-based selection
                if len(moves) > 0:
                    # Convert scores to probabilities using softmax with temperature
                    scores_array = np.array(scores, dtype=np.float32)
                    # Higher scores should have higher probability
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
                        best_move = moves[0]  # Fallback to best move
                else:
                    # Fallback if analysis fails
                    play_result = engine.play(
                        board,
                        chess.engine.Limit(depth=config.play_depth)
                    )
                    best_move = play_result.move
            else:
                # Use best move (original behavior)
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
            
            # Label position (potentially with deeper search)
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
                # Convert to centipawns from current player's perspective
                cp_score = score.white().score(mate_score=10000)
                if board.turn == chess.BLACK:
                    cp_score = -cp_score
                value = centipawns_to_value(cp_score, config.value_clip)
            else:
                value = 0.0
            
            # Encode board and move
            board_tensor = encode_board(board)
            policy_idx, policy_uci = encode_move(best_move)
            
            # Store sample
            sample = {
                'board': board_tensor,
                'policy_idx': policy_idx,
                'policy_uci': policy_uci,
                'value': value,
                'fen': board.fen(),
                'game_id': game_id
            }
            samples.append(sample)
            
            # Make the move
            board.push(best_move)
            move_count += 1
        
        engine.quit()
        logger.info(f"Game {game_id} complete: {len(samples)} positions, result: {board.result()}")
        
    except Exception as e:
        logger.error(f"Error generating game {game_id}: {e}")
        return []
    
    return samples


def generate_dataset_parallel(config: DatasetConfig) -> None:
    """
    Generate dataset using parallel workers.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config
    config_dict = {
        k: v for k, v in config.__dict__.items()
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    all_samples = []
    games_completed = 0
    
    logger.info(f"Starting dataset generation: {config.num_games} games with {config.parallel_workers} workers")
    start_time = time.time()
    
    if config.parallel_workers > 1:
        # Parallel generation
        with ProcessPoolExecutor(max_workers=config.parallel_workers) as executor:
            futures = {
                executor.submit(generate_single_game, config, i, config.stockfish_path): i
                for i in range(config.num_games)
            }
            
            for future in as_completed(futures):
                game_id = futures[future]
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                    games_completed += 1
                    
                    if games_completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = games_completed / elapsed
                        eta = (config.num_games - games_completed) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress: {games_completed}/{config.num_games} games "
                            f"({len(all_samples)} positions), "
                            f"{rate:.1f} games/sec, ETA: {eta/60:.1f} min"
                        )
                    
                    # Checkpoint
                    if games_completed % config.save_every == 0:
                        save_dataset(all_samples, output_dir, f"checkpoint_{games_completed}")
                        
                except Exception as e:
                    logger.error(f"Game {game_id} failed: {e}")
    else:
        # Sequential generation
        for i in range(config.num_games):
            samples = generate_single_game(config, i, config.stockfish_path)
            all_samples.extend(samples)
            games_completed += 1
            
            if games_completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = games_completed / elapsed
                eta = (config.num_games - games_completed) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {games_completed}/{config.num_games} games "
                    f"({len(all_samples)} positions), "
                    f"{rate:.1f} games/sec, ETA: {eta/60:.1f} min"
                )
            
            # Checkpoint
            if games_completed % config.save_every == 0:
                save_dataset(all_samples, output_dir, f"checkpoint_{games_completed}")
    
    # Final save
    save_dataset(all_samples, output_dir, "final")
    
    elapsed = time.time() - start_time
    logger.info(
        f"Dataset generation complete!\n"
        f"  Games: {games_completed}\n"
        f"  Positions: {len(all_samples)}\n"
        f"  Time: {elapsed/60:.1f} minutes\n"
        f"  Output: {output_dir}"
    )


def save_dataset(samples: List[Dict], output_dir: Path, name: str) -> None:
    """Save dataset to NPZ format."""
    if not samples:
        logger.warning("No samples to save")
        return
    
    # Convert to arrays
    boards = np.array([s['board'] for s in samples], dtype=np.float32)
    policy_indices = np.array([s['policy_idx'] for s in samples], dtype=np.int32)
    values = np.array([s['value'] for s in samples], dtype=np.float32)
    
    # Also save metadata
    fens = [s['fen'] for s in samples]
    policy_ucis = [s['policy_uci'] for s in samples]
    game_ids = np.array([s['game_id'] for s in samples], dtype=np.int32)
    
    # Save to NPZ
    output_path = output_dir / f"dataset_{name}.npz"
    np.savez_compressed(
        output_path,
        boards=boards,
        policy_indices=policy_indices,
        policy_ucis=policy_ucis,
        values=values,
        fens=fens,
        game_ids=game_ids
    )
    
    logger.info(f"Saved {len(samples)} samples to {output_path}")
    
    # Save statistics
    stats = {
        'num_samples': len(samples),
        'num_games': len(set(game_ids)),
        'board_shape': boards.shape,
        'value_mean': float(values.mean()),
        'value_std': float(values.std()),
        'value_min': float(values.min()),
        'value_max': float(values.max()),
    }
    
    with open(output_dir / f"stats_{name}.json", "w") as f:
        json.dump(stats, f, indent=2)


def main():
    """Example usage"""
    config = DatasetConfig(
        # Stockfish configuration
        stockfish_path="stockfish",  # Update this path!
        threads=1,
        hash_mb=256,
        
        # Search depth
        play_depth=10,
        label_depth=12,
        
        # Game generation - SCALE THESE UP for production!
        num_games=100,  # Start small for testing. Use 10,000+ for real training
        max_moves=100,  # Sufficient for most games
        
        # Diversity settings - CRITICAL for generalization
        opening_variety=True,
        opening_moves=10,  # 10-12 recommended (increases position diversity exponentially)
        stochastic_play=True,  # Prevents converging to same game lines
        stochastic_temperature=0.2,  # Controls randomness (0.1-0.3 works well)
        stochastic_rate=0.15,  # 15% of moves use stochastic selection
        
        # Output
        output_dir="chess_dataset",
        save_every=25,
        
        # Parallelization
        parallel_workers=1,  # Use 4-8+ on cloud/multi-core systems
    )
    
    generate_dataset_parallel(config)


if __name__ == "__main__":
    main()

