import os
import numpy as np # type: ignore
from datetime import datetime

import torch # type: ignore
from torchinfo import summary # type: ignore

# from torchmetrics import Accuracy # type: ignore

import mlflow.pytorch # type: ignore

from src.model import SampleNNClassifier, SimpleNN
from src.utils import create_dataloader, get_data_version

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # creates models directory
    MODELS_DIR = os.path.join(os.getcwd(), "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

class Training:
    def __init__(self, X : np.ndarry, y: np.ndarry, dvc_file_path : str, model : torch.nn.Module):
        self.X = X
        self.y = y
        self.MODEL_PATH = TrainingConfig()
        self.dvc_file_path = dvc_file_path
        
        self.model = model
    
    def training(self):
        """
        Train the model on the data.

        Args:
            None
        """
        # Create DataLoader (using same data for sample)
        train_dataloader = create_dataloader(self.X, self.y, self.model.batch_size)
        test_dataloader = create_dataloader(self.X, self.y, self.model.batch_size)

        # configure the location for MLflow to stores metadata 
        # mlflow.set_tracking_uri("file:/" + os.getcwd())

        # track experiment using date
        mlflow.set_experiment(datetime.today().strftime('%Y-%m-%d'))
        
        # Start an MLflow run
        with mlflow.start_run() as run:
            
            # define tracking parameters and data versions
            params = {
                "epochs": self.model.epochs,
                "learning_rate": self.model.learning_rate,
                "batch_size": self.model.batch_size,
                "loss_function": self.model.loss_criterion.__class__.__name__,
                "metric_function": self.model.metrics_fn.__class__.__name__,
                "optimizer": self.model.optimizer.__class__.__name__,
                "data_version" : get_data_version(self.dvc_file_path)
            }
            
            # Log training parameters.
            mlflow.log_params(params)

            # Log the latest run's model summary
            MODEL_SUMMARY_PATH = os.path.join(self.MODEL_PATH.MODELS_DIR, "model_summary_" + run.info.run_id + ".txt")
            with open(MODEL_SUMMARY_PATH, "w") as f:
                f.write(str(summary(self.model)))
            mlflow.log_artifact(MODEL_SUMMARY_PATH)
            
            for t in range(self.model.epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                
                # Train and evaluate the model
                self.train(train_dataloader, epoch=t)
                self.evaluate(test_dataloader, epoch=0)
            
            # Log the trained model
            mlflow.pytorch.log_model(self.model, "model")
    
    # Define the training loop for mlflow
    def train(self, dataloader, epoch):
        """Train the model on a single pass of the dataloader.

        Args:
            - dataloader (torch.utils.data.DataLoader): an instance containing the training data.
            - epoch (int): the current epoch number.
        """
        self.model.train()
        for batch, (inputs, labels) in enumerate(dataloader):
            
            loss, accuracy = self.model.training_step(batch, inputs, labels)
        
            # if batch % 100 == 0:
            loss, current = loss.item(), batch
            step = batch // 100 * (epoch + 1)
            mlflow.log_metric("loss", f"{loss:2f}", step=step)
            mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
            
            print(f"loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")

    def evaluate(self, dataloader, epoch):
        """Evaluate the model on a single pass of the dataloader.

        Args:
            - dataloader (torch.utils.data.DataLoader): an instance containing the eval data.
            - epoch (int): the current epoch number.
        """
        num_batches = len(dataloader)
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                loss, acc = self.model.validation_step(inputs, labels)
                eval_loss += loss
                eval_accuracy += acc

        eval_loss /= num_batches
        eval_accuracy /= num_batches
        mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
        mlflow.log_metric("eval_accuracy", f"{eval_accuracy:2f}", step=epoch)

        print(f"Eval metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n")
