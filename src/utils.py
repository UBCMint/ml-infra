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
    with open(dvc_file_path, 'r') as file:
        dvc_data = yaml.safe_load(file)
        return dvc_data['outs'][0]['md5'], dvc_data['outs'][0]['path']