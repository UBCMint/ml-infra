import torch
import torch.nn as nn
from torchmetrics import Accuracy
import torch.optim as optim

# Example NN for testing model versioning
class SimpleNN(nn.Module):
    """
    A PyTorch Simple NN model for classifying EEG data.

    Attributes:
        model (torch.nn.Module): The neural network architecture for classification.
        loss_criterion (torch.nn.Module): The loss function used during training.
        accuracy (torchmetrics.Metric): The accuracy metric for evaluating the model.
        learning_rate (float): The learning rate for the optimizer.
    """
    def __init__(
        self, input_size : int, num_classes : int, epochs : int, batch_size : int, learning_rate : float
    ):
        """
        Initializes the SimpleNN.

        Args:
            input_size (int): The size of each input sample.
            num_classes (int): The number of output classes for classification.
            epochs (int): The number of passes on the training data.
            batch_size (int): The size of batch for training and evaluating model
            learning_rate (float): The learning rate for the optimizer (default is 0.001).
        """
        super().__init__()
        self.input_size, self.num_classes = input_size, num_classes
        
        # Layers
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, num_classes)
        
        # Configure parameters
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        # Declare Loss Function
        self.loss_criterion = torch.nn.CrossEntropyLoss().to(device='cpu')
        # Declare metrics function
        self.metrics_fn = Accuracy(task="multiclass", num_classes=num_classes).to(device='cpu')
        # Instantiate optimizer
        self.configure_optimizers()

    # defines the flow of data from one layer to another in the neural network
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing EEG data.

        Returns:
            torch.Tensor: Model output after processing the input.
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, inputs, labels):
        """
        Performs a single training step.

        Args:
            batch (tuple): A batch of data containing input features and labels.

        Returns:
            torch.Tensor: The calculated loss for the batch.
            torch.Tensor: The calculated accuracy for the batch.
        """
        
        # Forward pass
        outputs = self(inputs)[:, -1, :]
        loss = self.loss_criterion(outputs, labels)
        acc = self.metrics_fn(outputs, labels)
        
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

        return loss, acc
    
    def validation_step(self, inputs, labels):
        """
        Performs a single validation step.

        Args:
            batch (tuple): A batch of validation data containing input features and labels.

        Returns:
            torch.Tensor: The calculated loss for model evaluation.
            torch.Tensor: The calculated accuracy for model evaluation.
        """
        outputs = self(inputs)[:, -1, :]
        val_loss = self.loss_criterion(outputs, labels).item()
        val_acc = self.metrics_fn(outputs, labels)
        
        return val_loss, val_acc
        
    def configure_optimizers(self):
        """
        Configures the optimizer for the training process.

        Returns:
            None
        """
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

# Inherits from the base class of all neural networks in PyTorch
class SampleNNClassifier(nn.Module):
    def __init__(
        self, input_size : int, num_classes : int, epochs : int, batch_size : int, learning_rate : float
    ):
        """
        Initializes the SampleNNClassifier.

        Args:
            input_size (int): The size of each input sample.
            num_classes (int): The number of output classes for classification.
            epochs (int): The number of passes on the training data.
            batch_size (int): The size of batch for training and evaluating model
            learning_rate (float): The learning rate for the optimizer (default is 0.001).
        """
        super().__init__()
        self.input_size, self.num_classes = input_size, num_classes
        
        # Layers
        self.fc1 = nn.Linear(input_size, 32)        # First fully connected layer
        self.fc2 = nn.Linear(32, 16)                # Second fully connected layer
        self.fc3 = nn.Linear(16, num_classes)       # Output layer
        self.dropout = nn.Dropout(0.1)              # Dropout Layer
        
        # Configure parameters
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        # Declare Loss Function
        self.loss_criterion = torch.nn.CrossEntropyLoss().to(device='cpu')
        # Declare metrics function
        self.metrics_fn = Accuracy(task="multiclass", num_classes=num_classes).to(device='cpu')
        # Instantiate optimizer
        self.configure_optimizers()

    # defines the flow of data from one layer to another in the neural network
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing EEG data.

        Returns:
            torch.Tensor: Model output after processing the input.
        """
        x = torch.relu(self.fc1(x))                 # Pass 1st layer and then apply relu
        x = self.dropout(x)                         # Dropout to output of first layer
        x = torch.relu(self.fc2(x))                 # Pass 2nd layer and then apply relu
        x = self.fc3(x)                             # Final output layer
        return x
    
    def training_step(self, batch, inputs, labels):
        """
        Performs a single training step.

        Args:
            batch (tuple): A batch of data containing input features and labels.

        Returns:
            torch.Tensor: The calculated loss for the batch.
            torch.Tensor: The calculated accuracy for the batch.
        """

        # Forward pass
        outputs = self(inputs)[:, -1, :]
        loss = self.loss_criterion(outputs, labels)
        acc = self.metrics_fn(outputs, labels)
        
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

        return loss, acc
    
    def validation_step(self, inputs, labels):
        """
        Performs a single validation step.

        Args:
            batch (tuple): A batch of validation data containing input features and labels.

        Returns:
            torch.Tensor: The calculated loss for model evaluation.
            torch.Tensor: The calculated accuracy for model evaluation.
        """
        outputs = self(inputs)[:, -1, :]
        val_loss = self.loss_criterion(outputs, labels).item()
        val_acc = self.metrics_fn(outputs, labels)
        
        return val_loss, val_acc
        
    def configure_optimizers(self):
        """
        Configures the optimizer for the training process.

        Returns:
            None
        """
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)