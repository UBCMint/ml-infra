import os
import numpy as np
from datetime import datetime

import torch
from torchinfo import summary

from torchmetrics import Accuracy

import mlflow.pytorch

from src.model import SampleNNClassifier, SimpleNN, TestSampleNNClassifier
from src.utils import create_dataloader, get_data_version

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # creates models directory
    MODELS_DIR = os.path.join(os.getcwd(), "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

class Training:
    def __init__(self, X, y, dvc_file_path):
        self.X = X
        self.y = y
        self.MODEL_PATH = TrainingConfig()
        self.dvc_file_path = dvc_file_path
    
    def training(self):
        """Train the model on the data.

        Args:
            None
        """
        input_size = self.X.shape[2]         # Adjust this to match the input size of your EEG data
        num_classes = len(np.unique(self.y)) # Adjust to the number of classes in your dataset
        
        ##############################################################################
        # MODEL 
        # Create Model Object
        model = SampleNNClassifier(
            input_size=input_size, 
            num_classes=num_classes, 
            epochs=10, 
            batch_size=16, 
            learning_rate=0.001
        )
        ##############################################################################
        
        # Create DataLoader (using same data for sample)
        train_dataloader = create_dataloader(self.X, self.y, model.batch_size)
        test_dataloader = create_dataloader(self.X, self.y, model.batch_size)

        # configure the location for MLflow to stores metadata 
        # mlflow.set_tracking_uri("file:/" + os.getcwd())

        # track experiment using date
        mlflow.set_experiment(datetime.today().strftime('%Y-%m-%d'))
        
        # Start an MLflow run
        with mlflow.start_run() as run:
            
            # define tracking parameters and data versions
            params = {
                "epochs": model.epochs,
                "learning_rate": model.learning_rate,
                "batch_size": model.batch_size,
                "loss_function": model.loss_criterion.__class__.__name__,
                "metric_function": model.metrics_fn.__class__.__name__,
                "optimizer": model.optimizer.__class__.__name__,
                "data_version" : get_data_version(self.dvc_file_path)
            }
            
            # Log training parameters.
            mlflow.log_params(params)

            # Log the latest run's model summary
            MODEL_SUMMARY_PATH = os.path.join(self.MODEL_PATH.MODELS_DIR, "model_summary_" + run.info.run_id + ".txt")
            with open(MODEL_SUMMARY_PATH, "w") as f:
                f.write(str(summary(model)))
            mlflow.log_artifact(MODEL_SUMMARY_PATH)
            
            for t in range(model.epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                
                # Train and evaluate the model
                train(model, train_dataloader, epoch=t)
                evaluate(model, test_dataloader, epoch=0)
            
            # Log the trained model
            mlflow.pytorch.log_model(model, "model")

# Define the training loop for mlflow
def train(model, dataloader, epoch):
    """Train the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        epoch: an integer, the current epoch number.
    """
    model.train()
    for batch in enumerate(dataloader):
        
        loss, accuracy = model.training_step(batch)
    
        # if batch % 100 == 0:
        loss, current = loss.item(), batch
        step = batch // 100 * (epoch + 1)
        mlflow.log_metric("loss", f"{loss:2f}", step=step)
        mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
        
        print(f"loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")

def evaluate(model, dataloader, epoch):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        epoch: an integer, the current epoch number.
    """
    num_batches = len(dataloader)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            loss, acc = model.validation_step(batch)
            eval_loss += loss
            eval_accuracy += acc

    eval_loss /= num_batches
    eval_accuracy /= num_batches
    mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric("eval_accuracy", f"{eval_accuracy:2f}", step=epoch)

    print(f"Eval metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n")
