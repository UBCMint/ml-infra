import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

# Load EEG Data
def load_eeg_data(file_path):
    data = np.load(file_path)
    X = data['X']  # Feature data
    y = data['y']  # Labels
    return X, y  

# Initialize dataset and dataloader
def create_dataloader(X, y, batch_size):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# log the specific data version (DVC hash and path) used for that experiment to MLflow.
def get_data_version(dvc_file_path):
    try:
        with open(dvc_file_path, 'r') as file:
            dvc_data = yaml.safe_load(file)
            return dvc_data['outs'][0]['md5'], dvc_data['outs'][0]['path']
    except Exception as e:
        print(f"Error getting DVC data version: {e}")

def check_dvc_status(data_path):
    """
    Check if there are changes in the data tracked by DVC.
    Args:
        data_path (str): The path to the data file or directory to be checked.
    Returns:
        bool: True if changes are detected, False otherwise.
    """
    try:
        # Check the status of DVC-tracked files
        status_output = subprocess.check_output(["dvc", "status", data_path], text=True)
        
        # If 'changes' or 'modified' is found in the output, there are changes
        if 'changed' in status_output or 'modified' in status_output:
            print("Data changes detected.")
            return True
        else:
            print("No changes in data.")
            return False
    except Exception as e:
        print(f"Error checking DVC status: {e}")
        return False

def update_dvc_data(data_path):
    """
    If changes are detected in the data, this function will:
    - Stage the data using `dvc add`.
    - Commit the changes to Git.
    - Push the new data version to the DVC remote.
    Args:
        data_path (str): The path to the data file or directory to be added.
    """
    try:
        # Stage new data version with DVC
        subprocess.run(["dvc", "add", data_path], check=True)
        print(f"Data added to DVC: {data_path}")

        # Commit the changes to Git
        subprocess.run(["git", "add", f"{data_path}.dvc", ".gitignore"], check=True)
        subprocess.run(["git", "commit", "-m", "Updated data"], check=True)
        print("Data versioning updated locally.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error updating DVC data: {e}")

def get_dvc_hash(data_path):
    """
    Get the current DVC hash (version) of the data file or directory.
    Args:
        data_path (str): The path to the data file or directory.
    Returns:
        str: The hash of the current data version.
    """
    try:
        # Run `dvc list` to retrieve the data hash
        hash_value = subprocess.getoutput(f'dvc get --show-url {data_path}')
        print(f"Current DVC hash for {data_path}: {hash_value}")
        return hash_value
    except Exception as e:
        print(f"Error retrieving DVC hash: {e}")
        return None