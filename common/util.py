import os
import ast
import random
import pickle
from typing import Dict, Any, Optional, List, Tuple, Union

import torch
import numpy as np
import importlib
import copy
import d4rl


device = None
logger = None
logger_model = None



def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device_and_logger(gpu_id, logger_ent, logger_mod):
    global device, logger, logger_model
    if gpu_id < 0 or torch.cuda.is_available() == False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(gpu_id))
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print("setting device:", device)
    logger = logger_ent
    logger_model = logger_mod



def relative_path_to_module_path(relative_path):
    path = relative_path.replace(".py", "").replace(os.path.sep,'.')
    return path


def load_config(config_path, update_args):
    default_config_path_elements = config_path.split(os.sep)
    default_config_path_elements[-1] = "default.py"
    default_config_path = os.path.join(*default_config_path_elements)
    default_args_module = importlib.import_module(relative_path_to_module_path(default_config_path))
    overwrite_args_module = importlib.import_module(relative_path_to_module_path(config_path))
    default_args_dict = getattr(default_args_module, 'default_args')
    args_dict = getattr(overwrite_args_module, 'overwrite_args')
    assert type(default_args_dict) == dict, "default args file should be default_args=\{...\}"
    assert type(args_dict) == dict, "args file should be default_args=\{...\}"

    #update args is tpule type, convert to dict type
    update_args_dict = {}
    for update_arg in update_args:
        key, val = update_arg.split("=")
        update_args_dict[key] = ast.literal_eval(val)
    
    #update env specific args to default 
    args_dict = merge_dict(default_args_dict, args_dict)
    default_args_dict = update_parameters(default_args_dict, update_args_dict)
    if 'common' in args_dict:
        for sub_key in args_dict:
            if type(args_dict[sub_key]) == dict:
                args_dict[sub_key] = merge_dict(args_dict[sub_key], default_args_dict['common'], "common")
    return args_dict


def merge_dict(source_dict, update_dict, ignored_dict_name=""):
    for key in update_dict:
        if key == ignored_dict_name:
            continue
        if key not in source_dict:
            #print("\033[32m new arg {}: {}\033[0m".format(key, update_dict[key]))
            source_dict[key] = update_dict[key]
        else:
            if type(update_dict[key]) == dict:
                source_dict[key] = merge_dict(source_dict[key], update_dict[key], ignored_dict_name)
            else:
                print("updated {} from {} to {}".format(key, source_dict[key], update_dict[key]))
                source_dict[key] = update_dict[key]
    return source_dict


def update_parameters(source_args, update_args):
    print("updating args", update_args)
    #command line overwriting case, decompose the path and overwrite the args
    for key_path in update_args:
        target_value = update_args[key_path]
        print("key:{}\tvalue:{}".format(key_path, target_value))
        source_args = overwrite_argument_from_path(source_args, key_path, target_value)
    return source_args


def overwrite_argument_from_path(source_dict, key_path, target_value):
    key_path = key_path.split("/")
    curr_dict = source_dict
    for key in key_path[:-1]:
        if not key in curr_dict:
            #illegal path
            return source_dict
        curr_dict = curr_dict[key]
    final_key = key_path[-1] 
    curr_dict[final_key] = target_value
    return source_dict


def second_to_time_str(remaining:int):
    dividers = [86400, 3600, 60, 1]
    names = ['day', 'hour', 'minute', 'second']
    results = []
    for d in dividers:
        re = int(np.floor(remaining / d))
        results.append(re)
        remaining -= re * d
    time_str = ""
    for re, name in zip(results, names):
        if re > 0 :
            time_str += "{} {}  ".format(re, name)
    return time_str


def load_dataset_with_validation_split(
    args,
    env=None,
    val_split_ratio: float = 0.2,
    test_split_ratio: float = 0.2,
    max_samples: Optional[int] = None,
    required_keys: List[str] = ['observations', 'actions', 'rewards', 'terminals']
) -> Dict[str, Any]:
    """
    Load dataset with automatic validation split creation when validation data doesn't exist.

    Args:
        args: Arguments object containing data_path and task information
        env: Environment object (needed for env.get_dataset() or abiomed datasets)
        val_split_ratio: Fraction of training data to use for validation (default: 0.2)
        max_samples: Maximum number of samples to load (useful for testing)
        required_keys: List of required keys in the dataset

    Returns:
        Dictionary containing:
        - 'train_data': Training dataset
        - 'val_data': Validation dataset
        - 'test_data': Test dataset (if available)
        - 'buffer_len': Total number of samples
        - 'data_info': Metadata about the dataset
    """

    dataset_info = {
        'source': None,
        'original_size': 0,
        'has_predefined_splits': False,
        'created_val_split': False
    }

    train_data = None
    val_data = None
    test_data = None

    # Case 1: Load from file path (pickle or npz)
    if hasattr(args, 'data_path') and args.data_path is not None:
        # TODO: make it up to date
        dataset_info['source'] = 'file'

        try:
            # Try pickle first
            with open(args.data_path, "rb") as f:
                dataset = pickle.load(f)
                print(f'Loaded pickle file: {args.data_path}')
                dataset_info['format'] = 'pickle'
        except:
            try:
                # Fallback to npz
                dataset = np.load(args.data_path)
                dataset = {k: dataset[k] for k in dataset.files}
                print(f'Loaded npz file: {args.data_path}')
                dataset_info['format'] = 'npz'
            except Exception as e:
                raise ValueError(f"Could not load dataset from {args.data_path}: {e}")
    else:
        # dataset = env.get_dataset()
        dataset = d4rl.qlearning_dataset(env)
        # Validate required keys
    missing_keys = [key for key in required_keys if key not in dataset]
    if missing_keys:
        available_keys = list(dataset.keys())
        print(f"Warning: Missing required keys {missing_keys}. Available keys: {available_keys}")

    # Apply max_samples limit if specified
    if max_samples is not None:
        dataset = {k: v[:max_samples] for k, v in dataset.items()}
        print(f'Limited dataset to {max_samples} samples')

    dataset_info['original_size'] = len(dataset['observations']) if 'observations' in dataset else len(list(dataset.values())[0])

    # Create train/val split since file datasets typically don't have predefined splits
    train_data, val_data = _create_train_val_split_dict(dataset, val_split_ratio + test_split_ratio)
    val_data, test_data = _create_train_val_split_dict(val_data, val_split_ratio / (val_split_ratio + test_split_ratio))
    dataset_info['created_val_split'] = True

    buffer_len = dataset_info['original_size']
    

    # Prepare return dictionary
    result = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'buffer_len': buffer_len,
        'data_info': dataset_info
    }

    # Print summary
    print(f"\nDataset loading summary:")
    print(f"  Source: {dataset_info['source']}")
    print(f"  Original size: {dataset_info['original_size']}")
    print(f"  Buffer length: {buffer_len}")
    if dataset_info['created_val_split']:
        print(f"  Created validation split: {val_split_ratio*100:.1f}% of training data")
    if dataset_info['has_predefined_splits']:
        print(f"  Using predefined train/val/test splits")

    return result



def _create_train_val_split_dict(
    dataset: Dict[str, np.ndarray],
    val_split_ratio: float
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split dataset into training and validation sets using temporal (sequential) split.

    Args:
        dataset: Dictionary containing dataset arrays
        val_split_ratio: Fraction of data to use for validation

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if dataset is None:
        raise ValueError("Dataset is empty")

    # Get dataset size from first key
    first_key = list(dataset.keys())[0]
    # print(first_key)
    total_size = len(dataset[first_key])
    # print(total_size)

    # Calculate split indices for temporal split
    # Train on earlier data, validate on later data
    val_size = int(total_size * val_split_ratio)
    train_size = total_size - val_size

    # Temporal split: no randomization, preserve temporal ordering
    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, total_size)

    # Split each array in the dataset
    train_dataset = {}
    val_dataset = {}
    for key, values in dataset.items():
        if isinstance(values, np.ndarray):
            train_dataset[key] = values[train_indices]
            val_dataset[key] = values[val_indices]
        else:
            # Handle non-numpy arrays (e.g., lists)
            values_array = np.array(values)
            # print( dataset.keys())
            train_dataset[key] = values_array[np.array(train_indices)]
            val_dataset[key] = values_array[np.array(val_indices)]

    print(f"Temporal split dataset: {train_size} train samples, {val_size} validation samples")

    return train_dataset, val_dataset


def validate_dataset_structure(
    dataset: Dict[str, Any],
    required_keys: List[str],
    min_samples: int = 1
) -> bool:
    """
    Validate that dataset has required structure and minimum samples.

    Args:
        dataset: Dataset dictionary to validate
        required_keys: List of keys that must be present
        min_samples: Minimum number of samples required

    Returns:
        True if dataset is valid, raises ValueError otherwise
    """
    if not isinstance(dataset, dict):
        raise ValueError("Dataset must be a dictionary")

    # Check required keys
    missing_keys = [key for key in required_keys if key not in dataset]
    if missing_keys:
        available_keys = list(dataset.keys())
        raise ValueError(f"Missing required keys: {missing_keys}. Available keys: {available_keys}")

    # Check that all arrays have the same length
    lengths = []
    for key in required_keys:
        if key in dataset:
            if hasattr(dataset[key], '__len__'):
                lengths.append(len(dataset[key]))
            else:
                raise ValueError(f"Dataset['{key}'] is not array-like")

    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent array lengths: {dict(zip(required_keys, lengths))}")

    # Check minimum samples
    if lengths and lengths[0] < min_samples:
        raise ValueError(f"Dataset has {lengths[0]} samples, but {min_samples} required")

    return True


