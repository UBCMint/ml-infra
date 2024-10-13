import torch
import torch.nn as nn

# Inherits from the base class of all neural networks in PyTorch
class SampleNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)        # First fully connected layer
        self.fc2 = nn.Linear(32, 16)                # Second fully connected layer
        self.fc3 = nn.Linear(16, num_classes)       # Output layer
        self.dropout = nn.Dropout(0.1)              # Dropout Layer
        
    # defines the flow of data from one layer to another in the neural network
    def forward(self, x):
        x = torch.relu(self.fc1(x))                 # Pass 1st layer and then apply relu
        x = self.dropout(x)                         # Dropout to output of first layer
        x = torch.relu(self.fc2(x))                 # Pass 2nd layer and then apply relu
        x = self.fc3(x)                             # Final output layer
        return x