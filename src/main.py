from .utils import chess_manager, GameContext
from chess import Move
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from pathlib import Path

# ===== Model Architecture =====

class ResBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class SimpleChessNet(nn.Module):
    """Simple CNN for chess position evaluation."""
    
    def __init__(self, num_filters: int = 256, num_res_blocks: int = 10):
        super().__init__()
        
        # Input: (batch, 14, 8, 8)
        self.conv_input = nn.Sequential(
            nn.Conv2d(14, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head (predicts best move)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096),  # 64 * 64 possible moves
        )
        
        # Value head (evaluates position)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x):
        x = self.conv_input(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value


# ===== Board Encoding Functions =====

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode chess board as neural network input tensor.
    
    Returns 14 planes of 8x8:
    - Planes 0-5: Current player's pieces (P, N, B, R, Q, K)
    - Planes 6-11: Opponent's pieces (P, N, B, R, Q, K)
    - Plane 12: Castling rights
    - Plane 13: En passant square, turn color, move count
    
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
        planes[12, 0, 7] = 1.0
    if board.has_queenside_castling_rights(board.turn):
        planes[12, 0, 0] = 1.0
    if board.has_kingside_castling_rights(not board.turn):
        planes[12, 7, 7] = 1.0
    if board.has_queenside_castling_rights(not board.turn):
        planes[12, 7, 0] = 1.0
    
    # Metadata plane (plane 13)
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        planes[13, ep_rank, ep_file] = 1.0
    
    planes[13, 0, 4] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13, 1, 4] = min(board.fullmove_number / 100.0, 1.0)
    
    return planes


def move_to_policy_index(move: chess.Move) -> int:
    """Convert a chess move to policy index (from_square * 64 + to_square)."""
    return move.from_square * 64 + move.to_square


def policy_index_to_move(policy_idx: int) -> tuple:
    """Convert policy index to (from_square, to_square)."""
    from_square = policy_idx // 64
    to_square = policy_idx % 64
    return from_square, to_square


# ===== Model Loading =====

# Load model once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to your trained model - update this path!
MODEL_PATH = Path(__file__).parent.parent / "dataset_generator" / "chess_model_final.pt"

# Create model with same architecture as training (adjust these if you used different values)
model = SimpleChessNet(num_filters=128, num_res_blocks=5)

# Load trained weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print(f"[OK] Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"[WARNING] Model file not found at {MODEL_PATH}")
    print("  Please update MODEL_PATH or train a model first.")
    model = None
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    model = None


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """Make a move using the trained neural network."""
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    if model is None:
        # Fallback to random if model not loaded
        print("Model not loaded, using random moves")
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(move_probs)
        return legal_moves[0]
    
    # Encode the board
    board_tensor = encode_board(ctx.board)
    board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(device)
    
    # Get model predictions
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
        policy_logits = policy_logits.squeeze(0)  # Remove batch dimension
    
    # Create a mapping of legal moves to their policy indices
    legal_move_indices = {}
    for move in legal_moves:
        policy_idx = move_to_policy_index(move)
        legal_move_indices[move] = policy_idx
    
    # Extract logits only for legal moves
    legal_logits = torch.tensor([
        policy_logits[policy_idx].item() 
        for policy_idx in legal_move_indices.values()
    ])
    
    # Convert to probabilities using softmax
    probabilities = F.softmax(legal_logits, dim=0).cpu().numpy()
    
    # Create probability dictionary for logging
    move_probs = {
        move: float(prob) 
        for move, prob in zip(legal_moves, probabilities)
    }
    ctx.logProbabilities(move_probs)
    
    # Select move (use argmax for deterministic, or sample for stochastic)
    # Using argmax for best move:
    best_idx = probabilities.argmax()
    selected_move = legal_moves[best_idx]
    
    # Optional: Print evaluation
    print(f"Position value: {value.item():.3f}, Selected: {selected_move.uci()}")
    
    return selected_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Reset for a new game."""
    # Model is stateless, so nothing to reset
    pass
