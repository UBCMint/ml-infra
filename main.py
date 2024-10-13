from src.data_ingestion import DataIngestion
from src.train import Training, load_eeg_data

# Execute the training process
if __name__ == "__main__":
    # Initialize DataIngestion object
    eeg_data = DataIngestion()
    
    # Load and ilter EEG data
    X, X_train, y = eeg_data.load_and_filter_eeg()
    
    # Save processed data
    eeg_data.save_eeg_data(X_train, y, eeg_data.dataIngestionConfig.RAW_DATA_PATH)
    
    # Load the saved raw data artifact
    X_train, y = load_eeg_data(eeg_data.dataIngestionConfig.RAW_DATA_PATH)
    
    # Perform training and evaluation
    trainer = Training(X_train, y, eeg_data.dataIngestionConfig.RAW_DATA_PATH + '.dvc')
    trainer.training()
    
    # Comment out to see dashboard
    # subprocess.run(["mlflow", "ui"], check=True)