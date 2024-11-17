from src.data_ingestion import DataIngestion, DataIngestionConfig
from src.train import Training
from src.utils import load_eeg_data, check_dvc_status, update_dvc_data

# Execute the training process
if __name__ == "__main__":
    # data replaced manually (ensure data is named "sample_raw_data.npz") and in a numpy file
    manual_ingestion = False
    
    # Initialize DataIngestion object with training time (tmin and tmax) for testing dvc
    train_tmin = 1.0
    train_tmax = 3.0
    
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
    
    # Perform training and evaluation
    DVC_FILE_PATH = DATA_PATH + '.dvc'
    trainer = Training(X_train, y, DVC_FILE_PATH)
    trainer.training()
    
    # Comment out to see mlflow dashboard
    # subprocess.run(["mlflow", "ui"], check=True)