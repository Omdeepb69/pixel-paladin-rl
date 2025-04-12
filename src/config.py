import pathlib
import torch

# --- Path Configurations ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
TENSORBOARD_LOG_DIR = LOG_DIR / "tensorboard"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)


# --- Model Parameters ---
MODEL_CONFIG = {
    "architecture": "CNN",  # Example: CNN, MLP
    "input_channels": 4,    # Number of stacked frames
    "output_actions": 6,    # Number of possible actions in the environment
    "learning_rate": 1e-4,
    "optimizer": "Adam",    # Example: Adam, RMSprop
    "loss_function": "MSELoss", # Example: MSELoss, SmoothL1Loss
    "hidden_units_cnn": [32, 64, 64], # Example CNN layer filters
    "kernel_sizes": [8, 4, 3],       # Example CNN kernel sizes
    "strides": [4, 2, 1],            # Example CNN strides
    "hidden_units_fc": [512],        # Example Fully Connected layer units
    "activation_function": "ReLU",   # Example: ReLU, Tanh
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dueling_dqn": True,             # Use Dueling DQN architecture
}


# --- Training Parameters ---
TRAINING_CONFIG = {
    "algorithm": "DQN",             # Example: DQN, PPO, A2C
    "total_timesteps": 1_000_000,   # Total steps for training
    "buffer_size": 100_000,         # Replay buffer size
    "batch_size": 32,               # Batch size for training updates
    "gamma": 0.99,                  # Discount factor
    "epsilon_start": 1.0,           # Initial exploration rate
    "epsilon_end": 0.01,            # Final exploration rate
    "epsilon_decay_steps": 250_000, # Steps over which epsilon decays
    "target_update_freq": 1000,     # Frequency of updating the target network (in steps)
    "learning_starts": 10_000,      # Number of steps before learning starts
    "train_freq": 4,                # Frequency of training updates (in steps)
    "gradient_steps": 1,            # Number of gradient steps per training update
    "seed": 42,                     # Random seed for reproducibility
    "log_interval": 1000,           # Log training progress every N steps
    "save_freq": 50_000,            # Save model checkpoint every N steps
    "eval_freq": 25_000,            # Evaluate model every N steps
    "num_eval_episodes": 10,        # Number of episodes for evaluation
    "max_grad_norm": 10,            # Max norm for gradient clipping
}


# --- Environment Configuration ---
ENV_CONFIG = {
    "env_id": "PixelPaladin-v0",    # Custom environment ID or Gym ID
    "screen_width": 84,
    "screen_height": 84,
    "grayscale": True,
    "frame_stack": 4,               # Number of frames to stack
    "frame_skip": 4,                # Number of frames to skip per action
    "reward_clipping": True,        # Clip rewards to [-1, 1]
    "terminal_on_life_loss": True,  # Treat losing a life as episode end during training
    "max_episode_steps": None,      # Maximum steps per episode (None for no limit)
    "render_mode": None,            # 'human', 'rgb_array', or None
}

# --- Evaluation Configuration ---
EVAL_CONFIG = {
    "model_path": CHECKPOINT_DIR / "best_model.zip", # Path to the model to evaluate
    "n_eval_episodes": 20,
    "render": True,
    "deterministic": True, # Use deterministic actions for evaluation
    "eval_log_path": LOG_DIR / "evaluations",
}
EVAL_CONFIG["eval_log_path"].mkdir(parents=True, exist_ok=True)


# --- General Project Info ---
PROJECT_NAME = "Pixel Paladin RL"
VERSION = "0.1.0"


# --- Helper Function (Optional) ---
def get_config_dict():
    """Returns a dictionary containing all configurations."""
    return {
        "paths": {
            "base_dir": str(BASE_DIR),
            "data_dir": str(DATA_DIR),
            "log_dir": str(LOG_DIR),
            "model_dir": str(MODEL_DIR),
            "checkpoint_dir": str(CHECKPOINT_DIR),
            "tensorboard_log_dir": str(TENSORBOARD_LOG_DIR),
        },
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "environment": ENV_CONFIG,
        "evaluation": {
            "model_path": str(EVAL_CONFIG["model_path"]),
            "n_eval_episodes": EVAL_CONFIG["n_eval_episodes"],
            "render": EVAL_CONFIG["render"],
            "deterministic": EVAL_CONFIG["deterministic"],
            "eval_log_path": str(EVAL_CONFIG["eval_log_path"]),
        },
        "project": {
            "name": PROJECT_NAME,
            "version": VERSION,
        }
    }

if __name__ == "__main__":
    # Example usage: Print a specific configuration value
    print(f"Using device: {MODEL_CONFIG['device']}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Total training timesteps: {TRAINING_CONFIG['total_timesteps']}")
    # print(get_config_dict()) # Uncomment to print all configs