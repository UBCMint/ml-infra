from src.data_ingestion import DataIngestion
from src.train import Training
from src.utils import load_eeg_data, check_dvc_status, update_dvc_data

# Execute the training process
if __name__ == "__main__":
    # Initialize DataIngestion object with 
    train_tmin = 1.0
    train_tmax = 2.0
    eeg_data = DataIngestion(train_tmin=train_tmin, train_tmax=train_tmax)
    DATA_PATH = eeg_data.dataIngestionConfig.RAW_DATA_PATH
    
    # Load and ilter EEG data
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
    trainer = Training(X_train, y, DATA_PATH+ '.dvc')
    trainer.training()
    
    # Comment out to see mlflow dashboard
    # subprocess.run(["mlflow", "ui"], check=True)