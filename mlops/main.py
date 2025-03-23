from src.data_ingestion import DataIngestion, DataIngestionConfig
from src.train import Training
from src.utils import load_eeg_data, check_dvc_status, update_dvc_data

import numpy as np

# import the models to be declared
from src.model import SampleNNClassifier, SimpleNN

# Execute the training process 
if __name__ == "__main__":
    # data replaced manually (ensure data is named "sample_raw_data.npz") and in a numpy file
    manual_ingestion = False
    
    # Initialize DataIngestion object with training time (tmin and tmax) for testing dvc
    train_tmin = 1.0
    train_tmax = 2.0
    
    eeg_data = None
    # DATA_PATH = None
    
    if manual_ingestion:
        config = DataIngestionConfig()
        DATA_PATH = config.RAW_DATA_PATH
    else:
        eeg_data = DataIngestion(train_tmin=train_tmin, train_tmax=train_tmax)
        DATA_PATH = eeg_data.dataIngestionConfig.RAW_DATA_PATH
        
        # Load and filter EEG data
        X, X_train, y = eeg_data.load_and_filter_eeg()
        
        # Save processed data
        eeg_data.save_eeg_data(X_train, y, DATA_PATH)
    
    # Check if the data has been changed
    if check_dvc_status(DATA_PATH):
        # If there are changes, update DVC and push to remote
        update_dvc_data(DATA_PATH)
    
    # Load the saved raw data artifact
    X_train, y = load_eeg_data(DATA_PATH)
    
    # Track the DVC file path
    DVC_FILE_PATH = DATA_PATH + '.dvc'
    
    input_size = X_train.shape[2]         # Adjust this to match the input size of your EEG data
    num_classes = len(np.unique(y))       # Adjust to the number of classes in your dataset
    
    ##############################################################################
        # MODELS
        # declare the models in list format that need to be run on the data
    models = [
        SampleNNClassifier(
            input_size=input_size, 
            num_classes=num_classes, 
            epochs=10, 
            batch_size=16, 
            learning_rate=0.001
        ),
        SimpleNN(
            input_size=input_size, 
            num_classes=num_classes, 
            epochs=10, 
            batch_size=16, 
            learning_rate=0.003
        )
    ]
    ##############################################################################
    
    # Run models on data
    for model in models:
        trainer = Training(X_train, y, DVC_FILE_PATH, model)
        trainer.training()
    
    # Comment out to see mlflow dashboard
    # subprocess.run(["mlflow", "ui"], check=True)