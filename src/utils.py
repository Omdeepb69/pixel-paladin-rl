```python
import os
import json
import pickle
import logging
import random
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        raise

def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    try:
        ensure_dir_exists(os.path.dirname(config_path))
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Error saving configuration to {config_path}: {e}")
        raise

def ensure_dir_exists(dir_path: str) -> None:
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            logging.debug(f"Created directory (or ensured it exists): {dir_path}")
        except OSError as e:
            logging.error(f"Error creating directory {dir_path}: {e}")
            raise

def save_data(data: Any, file_path: str) -> None:
    try:
        ensure_dir_exists(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise

def load_data(file_path: str) -> Any:
    if not os.path.exists(file_path):
        logging.warning(f"Data file not found: {file_path}. Returning None.")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Data successfully loaded from {file_path}")
        return data
    except EOFError:
        logging.error(f"Error loading data from {file_path}: File is empty or corrupted (EOFError). Returning None.")
        return None
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def save_q_table(q_table: Dict[Any, np.ndarray], file_path: str) -> None:
    save_data(q_table, file_path)

def load_q_table(file_path: str) -> Optional[Dict[Any, np.ndarray]]:
    data = load_data(file_path)
    if data is None:
        return None
    if isinstance(data, dict):
        # Basic check if values look like numpy arrays (can be more specific)
        if all(isinstance(v, np.ndarray) for v in data.values()):
             return data
        elif not data: # Allow empty dictionary
             return data
        else:
             logging.warning(f"Loaded dictionary from {file_path} contains non-numpy array values.")
             return data # Return anyway, let caller handle potential issues
    else:
        logging.error(f"Loaded data from {file_path} is not a dictionary (Q-table). Type: {type(data)}")
        raise TypeError("Loaded data is not of the expected type (Dict).")


def save_rewards(rewards_history: List[float], file_path: str) -> None:
    try:
        ensure_dir_exists(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(rewards_history, f)
        logging.info(f"Rewards history saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving rewards history to {file_path}: {e}")
        raise

def load_rewards(file_path: str) -> Optional[List[float]]:
    if not os.path.exists(file_path):
        logging.warning(f"Rewards file not found: {file_path}. Returning None.")
        return None
    try:
        with open(file_path, 'r') as f:
            rewards_history = json.load(f)
        if isinstance(rewards_history, list):
            logging.info(f"Rewards history loaded from {file_path}")
            return rewards_history
        else:
            logging.error(f"Data in {file_path} is not a list.")
            raise TypeError("Loaded data is not of the expected type (List).")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading rewards history from {file_path}: {e}")
        raise


def calculate_moving_average(data: List[Union[int, float]], window_size: int) -> np.ndarray:
    if not data or window_size <= 0:
        return np.array([])
    if window_size > len(data):
         window_size = len(data) # Calculate average over available data

    weights = np.repeat(1.0, window_size) / window_size
    moving_avg = np.convolve(np.array(data), weights, 'valid')

    # Pad the beginning with NaN to match the original length for plotting
    padding = np.full(window_size - 1, np.nan)
    return np.concatenate((padding, moving_avg))


def calculate_average_reward(rewards: List[float], window: int = 100) -> float:
    if not rewards:
        return 0.0
    actual_window = min(window, len(rewards))
    if actual_window == 0:
        return 0.0
    return np.mean(rewards[-actual_window:])


def plot_rewards(rewards_history: List[float],
                 title: str = "Rewards per Episode",
                 xlabel: str = "Episode",
                 ylabel: str = "Total Reward",
                 save_path: Optional[str] = None,
                 window_size: int = 100,
                 show_plot: bool = True) -> None:
    if not rewards_history:
        logging.warning("Cannot plot empty rewards history.")
        return

    episodes = np.arange(1, len(rewards_history) + 1)
    moving_avg = calculate_moving_average(rewards_history, window_size)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards_history, label='Reward per Episode', alpha=0.6, linewidth=1)

    # Plot moving average only where it's valid (not NaN)
    valid_indices = ~np.isnan(moving_avg)
    if np.any(valid_indices):
        plt.plot(episodes[valid_indices], moving_avg[valid_indices],
                 label=f'{window_size}-Episode Moving Average', color='red', linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        try:
            ensure_dir_exists(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300)
            logging.info(f"Plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving plot to {save_path}: {e}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = "pixel_paladin_rl.log", log_to_console: bool = True) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handlers = []
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_file:
        try:
            ensure_dir_exists(os.path.dirname(log_file))
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except Exception as e:
            # Log initial error to stderr if console logging is off but file logging fails
            if not log_to_console:
                import sys
                print(f"ERROR: Failed to set up file logging handler for {log_file}: {e}", file=sys.stderr)
            else:
                logging.error(f"Failed to set up file logging handler for {log_file}: {e}")

    # Remove existing handlers from the root logger before adding new ones
    # This prevents duplicate logs if setup_logging is called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=level, handlers=handlers)

    # Silence overly verbose libraries if necessary
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.info(f"Logging setup complete. Level: {log_level.upper()}. Console: {log_to_console}. File: {log_file if log_file else 'None'}.")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logging.debug("PyTorch random seed set.")
    except ImportError:
        logging.debug("PyTorch not found, skipping PyTorch seed setting.")
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Optional: For TF 1.x Graph-level seed
        # tf.compat.v1.set_random_seed(seed)
        logging.debug("TensorFlow random seed set.")
    except ImportError:
        logging.debug("TensorFlow not found, skipping TensorFlow seed setting.")
        pass

    logging.info(f"Global random seed set to {seed}")


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        logging.warning(f"Cannot normalize value: max_val ({max_val}) must be greater than min_val ({min_val}). Returning 0.")
        return 0.0
    return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)

def scale_value(normalized_value: float, min_val: float, max_val: float) -> float:
    if not (0.0 <= normalized_value <= 1.0):
         logging.warning(f"Input normalized_value ({normalized_value}) is outside the expected [0, 1] range.")
         normalized_value = np.clip(normalized_value, 0.0, 1.0)
    return normalized_value * (max_val - min_val) + min_val


if __name__ == "__main__":
    # Example usage block
    LOG_DIR = "logs"
    DATA_DIR = "data"
    PLOTS_DIR = "plots"
    CONFIG_FILE = os.path.join(DATA_DIR, "test_config.json")
    DATA_FILE = os.path.join(DATA_DIR, "test_data.pkl")
    REWARDS_FILE = os.path.join(DATA_DIR, "test_rewards.json")
    PLOT_FILE = os.path.join(PLOTS_DIR, "test_rewards_plot.png")
    LOG_FILE = os.path.join(LOG_DIR, "utils_test.log")

    # Ensure base directories exist for the test
    ensure_dir_exists(LOG_DIR)
    ensure_dir_exists(DATA_DIR)
    ensure_dir_exists(PLOTS_DIR)

    # Setup logging for the test run
    setup_logging(log_level="INFO", log_file=LOG_FILE, log_to_console=True)
    logging.info("--- Starting utils.py example usage ---")

    # --- Config Example ---
    logging.info("Testing configuration management...")
    test_config = {"agent_type": "Q-Learning", "learning_rate": 0.1, "gamma": 0.95, "episodes": 500, "epsilon_decay": 0.995}
    save_config(test_config, CONFIG_FILE)
    loaded_config = load_config(CONFIG_FILE)
    logging.info(f"Loaded config: {loaded_config}")
    assert test_config == loaded_config, "Config load/save failed!"
    logging.info("Configuration management test PASSED.")

    # --- File Ops Example ---
    logging.info("Testing file operations (generic data)...")
    test_data = {"metrics": {"accuracy": 0.95, "loss": 0.1}, "model_params": [1.0, -0.5, 0.01]}
    save_data(test_data, DATA_FILE)
    loaded_data = load_data(DATA_FILE)
    logging.info(f"Loaded generic data: {loaded_data}")
    assert test_data == loaded_data, "