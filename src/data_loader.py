import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Any, Optional, Union
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
STATE_FILE = 'states.pkl'
ACTION_FILE = 'actions.pkl'
REWARD_FILE = 'rewards.pkl'
NEXT_STATE_FILE = 'next_states.pkl'
DONE_FILE = 'dones.pkl'
DATA_KEYS = ['states', 'actions', 'rewards', 'next_states', 'dones']

# --- Data Loading ---

def _load_pickle_data(file_path: str) -> Optional[Any]:
    """Loads data from a pickle file."""
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def load_experience_data(data_dir: str) -> Optional[Dict[str, List[Any]]]:
    """
    Loads pre-recorded experience data (lists of transitions) from pickle files
    stored in the specified directory.

    Assumes each file contains a list corresponding to transitions.
    For example, states.pkl contains [state_1, state_2, ..., state_N].

    Args:
        data_dir: The directory containing the data files
                  (states.pkl, actions.pkl, etc.).

    Returns:
        A dictionary containing the loaded data lists, or None if loading fails.
        Keys are 'states', 'actions', 'rewards', 'next_states', 'dones'.
    """
    if not os.path.isdir(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        return None

    data = {}
    file_map = {
        'states': STATE_FILE,
        'actions': ACTION_FILE,
        'rewards': REWARD_FILE,
        'next_states': NEXT_STATE_FILE,
        'dones': DONE_FILE,
    }

    loaded_lengths = []
    for key, filename in file_map.items():
        file_path = os.path.join(data_dir, filename)
        loaded_list = _load_pickle_data(file_path)
        if loaded_list is None:
            logging.error(f"Failed to load {key} data. Aborting.")
            return None
        if not isinstance(loaded_list, list):
             logging.error(f"Expected a list in {filename}, but got {type(loaded_list)}. Aborting.")
             return None
        data[key] = loaded_list
        loaded_lengths.append(len(loaded_list))

    # Validate data consistency
    if len(set(loaded_lengths)) > 1:
        logging.error(f"Inconsistent data lengths found across files: {loaded_lengths}")
        logging.error("Ensure all data files (states, actions, etc.) have the same number of entries.")
        return None
    if not loaded_lengths:
         logging.warning(f"No data loaded from directory {data_dir}.")
         return None # Or return empty dict depending on desired behavior

    logging.info(f"Successfully loaded all experience data components from {data_dir}. Total transitions: {loaded_lengths[0]}")
    return data

# --- Data Preprocessing and Cleaning ---

def clean_data(data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Performs basic cleaning on the loaded data.
    Currently, this function is a placeholder. Implement specific cleaning
    logic based on the nature of your data (e.g., remove outliers, handle NaNs).

    Args:
        data: The dictionary containing loaded data lists.

    Returns:
        The cleaned data dictionary.
    """
    # Example: Remove transitions where reward is an unexpected type (if needed)
    # initial_len = len(data['rewards'])
    # valid_indices = [i for i, r in enumerate(data['rewards']) if isinstance(r, (int, float))]
    # if len(valid_indices) < initial_len:
    #     logging.warning(f"Removing {initial_len - len(valid_indices)} transitions due to invalid reward types.")
    #     cleaned_data = {}
    #     for key in DATA_KEYS:
    #         cleaned_data[key] = [data[key][i] for i in valid_indices]
    #     return cleaned_data

    logging.info("Data cleaning step completed (currently placeholder).")
    # In this basic version, we assume data is already clean
    return data

# --- Data Transformation and Feature Engineering ---

def transform_states_to_numpy(
    states: List[Any],
    next_states: List[Any],
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts lists of states and next_states to NumPy arrays.
    Assumes states are suitable for direct conversion (e.g., lists, tuples, NumPy arrays).

    Args:
        states: List of state representations.
        next_states: List of next state representations.
        dtype: The desired NumPy data type for the arrays.

    Returns:
        A tuple containing (numpy_states, numpy_next_states).
    """
    try:
        np_states = np.array(states, dtype=dtype)
        np_next_states = np.array(next_states, dtype=dtype)
        logging.info(f"Transformed states to NumPy arrays with shape: {np_states.shape}")
        return np_states, np_next_states
    except Exception as e:
        logging.error(f"Error converting states to NumPy arrays: {e}")
        raise ValueError("Could not convert states/next_states lists to NumPy arrays. Check state format.") from e

def normalize_states(
    states: np.ndarray,
    next_states: np.ndarray,
    scale_factor: float = 255.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes state arrays, typically used for image-based states.
    Assumes states are pixel values in the range [0, scale_factor].
    Scales them to [0.0, 1.0].

    Args:
        states: NumPy array of states.
        next_states: NumPy array of next states.
        scale_factor: The maximum value to scale by (e.g., 255 for RGB images).

    Returns:
        A tuple containing the normalized (states, next_states).
    """
    if scale_factor <= 0:
        logging.error("Scale factor must be positive.")
        raise ValueError("Scale factor must be positive.")

    normalized_states = states / scale_factor
    normalized_next_states = next_states / scale_factor
    logging.info(f"Normalized states using scale factor: {scale_factor}")
    return normalized_states, normalized_next_states

def transform_data(
    data: Dict[str, List[Any]],
    normalize: bool = True,
    scale_factor: float = 255.0
) -> Dict[str, np.ndarray]:
    """
    Applies transformations to the raw data dictionary.
    Converts lists to NumPy arrays and optionally normalizes states.

    Args:
        data: The dictionary containing loaded data lists.
        normalize: Whether to normalize the state arrays.
        scale_factor: The scale factor to use for normalization.

    Returns:
        A dictionary containing the transformed data as NumPy arrays.
    """
    transformed_data = {}

    # Convert states and next_states first
    states_np, next_states_np = transform_states_to_numpy(data['states'], data['next_states'])

    if normalize:
        states_np, next_states_np = normalize_states(states_np, next_states_np, scale_factor)

    transformed_data['states'] = states_np
    transformed_data['next_states'] = next_states_np

    # Convert other components to NumPy arrays
    try:
        transformed_data['actions'] = np.array(data['actions']) # Consider dtype based on action space
        transformed_data['rewards'] = np.array(data['rewards'], dtype=np.float32)
        transformed_data['dones'] = np.array(data['dones'], dtype=np.bool_) # Or np.uint8
        logging.info("Transformed actions, rewards, dones to NumPy arrays.")
    except Exception as e:
        logging.error(f"Error converting actions/rewards/dones to NumPy arrays: {e}")
        raise ValueError("Could not convert actions/rewards/dones lists to NumPy arrays.") from e

    return transformed_data


# --- Data Splitting ---

def split_data(
    data: Dict[str, np.ndarray],
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: Optional[int] = 42,
    shuffle: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        data: Dictionary of NumPy arrays ('states', 'actions', etc.).
        test_size: Proportion of the dataset to include in the test split.
        validation_size: Proportion of the dataset to include in the validation split.
        random_state: Controls the shuffling applied to the data before splitting.
        shuffle: Whether to shuffle the data before splitting.

    Returns:
        A dictionary containing the splits: {'train': {...}, 'validation': {...}, 'test': {...}}.
        Each inner dictionary holds the data arrays for that split.
        Returns an empty dictionary if splitting fails or data is insufficient.
    """
    num_samples = data['states'].shape[0]
    if num_samples == 0:
        logging.error("Cannot split data: dataset is empty.")
        return {}

    if not (0 < test_size < 1 and 0 <= validation_size < 1 and (test_size + validation_size) < 1):
        logging.error(f"Invalid split sizes: test_size={test_size}, validation_size={validation_size}. Must be between 0 and 1, and sum < 1.")
        return {}

    indices = np.arange(num_samples)

    # Split into train+validation and test
    try:
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
    except ValueError as e:
         logging.error(f"Error during initial train/test split: {e}. Check if dataset size is sufficient for the split.")
         return {}


    train_indices = train_val_indices
    val_indices = np.array([], dtype=int) # Initialize validation indices as empty

    # Split train+validation into train and validation if validation_size > 0
    if validation_size > 0 and len(train_val_indices) > 0:
        # Adjust validation size relative to the train_val set
        val_size_adjusted = validation_size / (1.0 - test_size)
        if val_size_adjusted >= 1.0:
             logging.warning(f"Validation size ({validation_size}) is too large relative to remaining data after test split. Using all remaining non-test data for training.")
             # train_indices are already set correctly
        else:
            try:
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=val_size_adjusted,
                    random_state=random_state,
                    shuffle=shuffle # Shuffle within train/val split as well if desired
                )
            except ValueError as e:
                 logging.error(f"Error during train/validation split: {e}. Check if train+validation set size is sufficient.")
                 # Proceed without validation set if split fails
                 logging.warning("Proceeding without a validation set due to split error.")
                 val_indices = np.array([], dtype=int)
                 # train_indices remain as train_val_indices

    splits = {'train': {}, 'validation': {}, 'test': {}}

    for key in data:
        splits['train'][key] = data[key][train_indices]
        splits['validation'][key] = data[key][val_indices] if len(val_indices) > 0 else np.array([]) # Handle empty validation set
        splits['test'][key] = data[key][test_indices]

    logging.info(f"Data split completed:")
    logging.info(f"  Train size: {len(train_indices)}")
    logging.info(f"  Validation size: {len(val_indices)}")
    logging.info(f"  Test size: {len(test_indices)}")

    # Check if validation set ended up empty and remove if so
    if len(val_indices) == 0:
        del splits['validation']
        logging.info("  Validation set is empty.")


    return splits

# --- Utility Function to Generate Dummy Data ---

def generate_dummy_data(
    save_dir: str,
    num_samples: int = 1000,
    state_shape: Tuple[int, ...] = (4,),
    num_actions: int = 2
) -> None:
    """Generates dummy experience data and saves it to files."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate simple random data for demonstration
    # States could be simple vectors or image-like (adjust state_shape)
    # Example: state_shape=(64, 64, 3) for images
    is_image_like = len(state_shape) > 1

    if is_image_like:
        # Assume pixel values 0-255
        states = [np.random.randint(0, 256, size=state_shape, dtype=np.uint8) for _ in range(num_samples)]
        next_states = [np.random.randint(0, 256, size=state_shape, dtype=np.uint8) for _ in range(num_samples)]
    else:
        # Assume feature vectors
        states = [np.random.rand(*state_shape).astype(np.float32) for _ in range(num_samples)]
        next_states = [np.random.rand(*state_shape).astype(np.float32) for _ in range(num_samples)]

    actions = [np.random.randint(0, num_actions) for _ in range(num_samples)]
    rewards = [np.random.rand() * 10 - 5 for _ in range(num_samples)]  # Rewards between -5 and 5
    dones = [np.random.rand() > 0.9 for _ in range(num_samples)] # ~10% done rate

    data_to_save = {
        STATE_FILE: states,
        ACTION_FILE: actions,
        REWARD_FILE: rewards,
        NEXT_STATE_FILE: next_states,
        DONE_FILE: dones,
    }

    for filename, data_list in data_to_save.items():
        file_path = os.path.join(save_dir, filename)
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data_list, f)
            logging.info(f"Saved dummy data to {file_path}")
        except Exception as e:
            logging.error(f"Error saving dummy data to {file_path}: {e}")

# --- Main Execution Example ---

if __name__ == "__main__":
    logging.info("Starting data_loader.py example execution.")

    # 1. Define parameters
    DUMMY_DATA_DIR = 'dummy_rl_data'
    NUM_SAMPLES = 5000
    # STATE_SHAPE = (64, 64, 1) # Example: Grayscale image state
    STATE_SHAPE = (8,)       # Example: Feature vector state
    NUM_ACTIONS = 4
    NORMALIZE = True # Set to False if states are not pixel-like or already normalized
    SCALE_FACTOR = 1.0 if not NORMALIZE or len(STATE_SHAPE) == 1 else 255.0 # Adjust scale factor based on state type

    # 2. Generate dummy data (only needed for demonstration)
    logging.info(f"Generating {NUM_SAMPLES} dummy data samples...")
    generate_dummy_data(DUMMY_DATA_DIR, NUM_SAMPLES, STATE_SHAPE, NUM_ACTIONS)

    # 3. Load the data
    logging.info(f"Loading data from {DUMMY_DATA_DIR}...")
    raw_data = load_experience_data(DUMMY_DATA_DIR)

    if raw_data:
        # 4. Clean the data (placeholder step)
        cleaned_data = clean_data(raw_data)

        # 5. Transform the data
        logging.info("Transforming data (converting to NumPy, normalizing)...")
        try:
            transformed_data = transform_data(
                cleaned_data,
                normalize=NORMALIZE,
                scale_factor=SCALE_FACTOR
            )

            # Print shapes after transformation
            logging.info("Data shapes after transformation:")
            for key, arr in transformed_data.items():
                logging.info(f"  {key}: {arr.shape}, dtype: {arr.dtype}")

            # 6. Split the data
            logging.info("Splitting data into train, validation, and test sets...")
            data_splits = split_data(
                transformed_data,
                test_size=0.2,
                validation_size=0.1,
                random_state=42
            )

            if data_splits:
                # Accessing split data examples:
                if 'train' in data_splits:
                    train_states = data_splits['train']['states']
                    logging.info(f"Train states shape: {train_states.shape}")
                    # Example: Get first 5 training actions
                    # logging.info(f"First 5 train actions: {data_splits['train']['actions'][:5]}")

                if 'validation' in data_splits and data_splits['validation']['states'].shape[0] > 0:
                    validation_rewards = data_splits['validation']['rewards']
                    logging.info(f"Validation rewards shape: {validation_rewards.shape}")
                elif 'validation' in data_splits:
                     logging.info("Validation set is present but empty.")
                else:
                     logging.info("No validation set was created.")


                if 'test' in data_splits:
                    test_dones = data_splits['test']['dones']
                    logging.info(f"Test dones shape: {test_dones.shape}")

                logging.info("Data loading, processing, and splitting finished successfully.")
            else:
                logging.error("Data splitting failed.")

        except ValueError as e:
            logging.error(f"An error occurred during data transformation: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during processing: {e}")
    else:
        logging.error("Failed to load initial data. Exiting.")

    logging.info("data_loader.py example execution finished.")