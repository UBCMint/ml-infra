# MINT MOSS Infrastructure and MLOps 

Developing infrastructure and MLOps tools for MINT MOSS.

## Setup and Installation
### Prerequisites
* Python 3.12
* Latest [Anaconda Distribution](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) Installed
    * **NOTE:** If unsure which one to install, see [this](https://docs.anaconda.com/distro-or-miniconda/).

### Set up Workspace

#### Fork this repository:

* Navigate to the GitHub repository.
* Click on the "Fork" button in the top-right corner.
* Clone the forked repository to your local machine and change working directory:
 ```
$ git clone https://github.com/UBCMint/ml-infra.git
$ cd ml-infra
```

#### Create and Activate Conda Environment

```
$ conda create -p venv python=3.12 -y
$ conda activate venv/
```

#### Install the required packages and dependencies:

```
$ pip install -r requirements.txt
```

## Workflow

### Data Ingestion

The data can be ingested into the system manually or through an script that downloads the data from an MNE dataset.

### Model Workflow

The model should be declared in `model.py` file as a class.

```
class DocNN(nn.Module):
    """
    A PyTorch Simple NN model for classifying EEG data.

    Attributes:
        model (torch.nn.Module): The neural network architecture for classification.
        loss_criterion (torch.nn.Module): The loss function used during training.
        accuracy (torchmetrics.Metric): The accuracy metric for evaluating the model.
        learning_rate (float): The learning rate for the optimizer.
    """
    def __init__(self, input_size, num_classes, epochs, batch_size, learning_rate):
        """
        Initializes the SimpleNN.

        Args:
            input_size (int): The size of each input sample.
            num_classes (int): The number of output classes for classification.
            epochs (int): The number of passes on the training data.
            batch_size (int): The size of batch for training and evaluating model
            learning_rate (float): The learning rate for the optimizer (default is 0.001).
        """
        super().__init__(input_size, num_classes)
        self.input_size, self.num_classes = input_size, num_classes
        
        # Layers
        super().__init__()
        <DECLARE-ARCHITECTURE>
        
        # Configure parameters
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        # Declare Loss Function
        self.loss_criterion = <LOSS-FUNCTION>.to(device='cpu')
        # Declare metrics function
        self.metrics_fn = <METRICS-FUNCTION>.to(device='cpu')
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
        <FORWARD-FLOW>
        return x
    
    def training_step(self, batch):
        """
        Performs a single training step.

        Args:
            batch (tuple): A batch of data containing input features and labels.

        Returns:
            torch.Tensor: The calculated loss for the batch.
            torch.Tensor: The calculated accuracy for the batch.
        """
        inputs, labels = batch
        
        # Forward pass
        outputs = self(inputs)[:, -1, :]
        loss = self.loss_criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

        return loss, acc
    
    def validation_step(self, batch):
        """
        Performs a single validation step.

        Args:
            batch (tuple): A batch of validation data containing input features and labels.

        Returns:
            torch.Tensor: The calculated loss for model evaluation.
            torch.Tensor: The calculated accuracy for model evaluation.
        """
        inputs, labels = batch
        outputs = self(inputs)[:, -1, :]
        val_loss = self.loss_criterion(outputs, labels).item()
        val_acc = self.accuracy(outputs, labels)
        
        return val_loss, val_acc
        
    def configure_optimizers(self):
        """
        Configures the optimizer for the training process.

        Returns:
            None
        """
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
```

### MLflow

#### Running the script for ML jobs

The script uses the MLflow Tracking API. For instance, run from the root directory of your workspace:
```
python main.py
```
This program will use [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html), which logs tracking data in ./mlruns. This can then be viewed with the Tracking UI.

#### Launching the Tracking UI

The MLflow Tracking UI will show runs logged in ./mlruns at http://localhost:5000. Start it with:
```
mlflow ui
```

#### Register a Model

Use the MLflow Model Registry UI to effectively manage and organize machine learning models.

Follow the steps in [Mlflow UI Workflow](https://mlflow.org/docs/latest/model-registry.html#ui-workflow) to register models in MLflow Registry.

**NOTES:** 
- To simulate model versioning, the [model.py file](https://github.com/UBCMint/ml-infra/blob/main/src/model.py) includes two models **SampleNNClassifier** and **SimpleNN**. The model summary is stored in the models directory as well logged as a artifact using mlflow.
- In [train.py](https://github.com/UBCMint/ml-infra/blob/main/src/train.py), make sure to create the model object, define loss function etc

### DVC

#### Tracking data

Working inside an initialized project directory, use the `add dvc` command to start tracking the dataset file ( with path <DATASET-FILE-PATH>, directory <DATASET-FILE-DIR> and name <DATASET-FILE-NAME>)

```
$ dvc add <DATASET-FILE-NAME>
```

DVC stores information about the added file in a special .dvc file named  <DATASET-FILE-NAME>.dvc. This small, human-readable metadata file acts as a placeholder for the original data for the purpose of Git tracking. This file contains a 'md5' hash value that uniquely identifies a dataset. 

Next, run the following commands to track changes in Git:
```
$ git add  <DATASET-FILE-NAME>.dvc <DATASET-FILE-DIR>/.gitignore
$ git commit -m <COMMIT-MESSAGE>
```

The data version or file can can be found be in the <.../ml-infra/.dvc/cache/files/md5>.