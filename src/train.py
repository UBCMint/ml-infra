import os
import numpy as np
from datetime import datetime
import yaml

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

from torchmetrics import Accuracy

import mlflow.pytorch

from src.data_ingestion import DataIngestion
from src.model import SampleNNClassifier

# Load EEG Data
def load_eeg_data(file_path):
    data = np.load(file_path)
    X = data['X']  # Feature data
    y = data['y']  # Labels
    return X, y  

# Initialize dataset and dataloader
def create_dataloader(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    batch_size = 16 # Adjust according to the dataset your system capabilities
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# log the specific data version (DVC hash and path) used for that experiment to MLflow.
def get_data_version(dvc_file_path):
    with open(dvc_file_path, 'r') as file:
        dvc_data = yaml.safe_load(file)
        return dvc_data['outs'][0]['md5'], dvc_data['outs'][0]['path']

# Define the training loop for mlflow
def train(model, dataloader, loss_fn, metrics_fn, optimizer, epoch):
    """Train the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
        epoch: an integer, the current epoch number.
    """
    model.train()
    for batch, (inputs, labels) in enumerate(dataloader):
        
        # Forward pass
        outputs = model(inputs)[:, -1, :]
        loss = loss_fn(outputs, labels)
        accuracy = metrics_fn(outputs, labels)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
        # if batch % 100 == 0:
        loss, current = loss.item(), batch
        step = batch // 100 * (epoch + 1)
        mlflow.log_metric("loss", f"{loss:2f}", step=step)
        mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
        print(f"loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")

def evaluate(model, dataloader, loss_fn, metrics_fn, epoch):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        epoch: an integer, the current epoch number.
    """
    num_batches = len(dataloader)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)[:, -1, :]
            eval_loss += loss_fn(outputs, labels).item()
            eval_accuracy += metrics_fn(outputs, labels)

    eval_loss /= num_batches
    eval_accuracy /= num_batches
    mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric("eval_accuracy", f"{eval_accuracy:2f}", step=epoch)

    print(f"Eval metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n")

class Training:
    def __init__(self, X, y, dvc_file_path):
        self.X = X
        self.y = y
        self.dvc_file_path = dvc_file_path
    
    def training(self):
        input_size = self.X.shape[2]         # Adjust this to match the input size of your EEG data
        num_classes = len(np.unique(self.y)) # Adjust to the number of classes in your dataset
        
        # Create DataLoader (using same data for sample)
        train_dataloader = create_dataloader(self.X, self.y)
        test_dataloader = create_dataloader(self.X, self.y)
        
        # define training hyperparameters, create model, declare loss function and instantiate optimizer.
        epochs = 10
        model = SampleNNClassifier(input_size=input_size, num_classes=num_classes)          # Using sample model from the nn.Module
        loss_criterion = torch.nn.CrossEntropyLoss().to(device='cpu')                       # Using criterion from nn.Module
        metrics_fn = Accuracy(task="multiclass", num_classes=num_classes).to(device='cpu')  # Using accuracy for multiclass 
        optimizer = optim.Adam(model.parameters(), lr=0.001)                                # Using adam optimizers
        
        # track experiment using date
        mlflow.set_experiment(datetime.today().strftime('%Y-%m-%d'))
        
        # Start an MLflow run
        with mlflow.start_run() as run:
            
            # define tracking parameters and data versions
            params = {
                "epochs": epochs,
                "learning_rate": 0.001,
                "batch_size": 64,
                "loss_function": loss_criterion.__class__.__name__,
                "metric_function": metrics_fn.__class__.__name__,
                "optimizer": "Adam",
                "data_version" : get_data_version(self.dvc_file_path)
            }
            
            # Log training parameters.
            mlflow.log_params(params)

            # Log the latest run's model summary
            with open("model_summary.txt", "w") as f:
                f.write(str(summary(model)))
            mlflow.log_artifact("model_summary.txt")
            
            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                
                # Train and evaluate the model
                train(model, train_dataloader, loss_criterion, metrics_fn, optimizer, epoch=t)
                evaluate(model, test_dataloader, loss_criterion, metrics_fn, epoch=0)
            
            # Log the trained model
            mlflow.pytorch.log_model(model, "model")