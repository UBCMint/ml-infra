# #############################################################################
# Followed MINT's Deep Learning Team's tutorial
# ---  
#   This Notebook largely takes from (https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html#ex-decoding-csp-eeg):
#   > Original Authors: Martin Billinger <martin.billinger@tugraz.at>
#   > License: BSD-3-Clause
#   > Copyright the MNE-Python contributors.
import os
import numpy as np

from mne import Epochs, pick_types
from mne.channels import make_standard_montage # type : ignore
from mne.datasets import eegbci # type : ignore
from mne.io import concatenate_raws, read_raw_edf # type : ignore

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    # creates mne data directory
    MNE_DATA_DIR = os.path.join(os.getcwd(), "mne")
    os.makedirs(MNE_DATA_DIR, exist_ok=True)

    # creates data directory
    DATA_DIR = os.path.join(os.getcwd(), "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    #  creates raw data directory
    RAW_DATA_DIR : str = os.path.join(DATA_DIR, "raw")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # # creates processed data directory
    # PROCESSED_DATA_DIR : str = os.path.join(DATA_DIR, "processed")
    # os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    RAW_DATA_PATH : str = os.path.join(RAW_DATA_DIR, "sample_raw_data.npz")

class DataIngestion:
    def __init__(self, train_tmin, train_tmax):
        self.dataIngestionConfig = DataIngestionConfig()
        
        # sample values to test data versioning
        self.train_tmin = train_tmin
        self.train_tmax = train_tmax

    def load_and_filter_eeg(self, subject=1, runs=[6, 10, 14], tmin=-1.0, tmax=4.0):
        """
        Load and filter EEG data for motor imagery task (hands vs. feet).
        """
        try:
            raw_fnames = eegbci.load_data(subject, runs, path=self.dataIngestionConfig.MNE_DATA_DIR)
            raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
            
            # Standardize the data and set montage (standard electrode configuration)
            eegbci.standardize(raw)
            montage = make_standard_montage("standard_1005")
            raw.set_montage(montage)
            raw.annotations.rename(dict(T1="hands", T2="feet"))
            
            # Set EEG reference and apply a band-pass filter (7-30 Hz)
            raw.set_eeg_reference(projection=True)
            raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
            
            # Select EEG channels
            picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
            
            # Create epochs (extract segments of data) for event-related study design
            epochs = Epochs(
                raw,
                event_id={"hands": 2, "feet": 3},
                tmin=tmin,
                tmax=tmax,
                proj=True,     
                picks=picks,
                baseline=None,
                preload=True,
            )
            
            # Crop training data to avoid classification based on early evoked responses
            epochs_train = epochs.copy().crop(tmin=self.train_tmin, tmax=self.train_tmax)
            
            # Labels: 0 for feet, 1 for hands
            labels = epochs.events[:, -1] - 2
            
            return epochs.get_data(copy=False), epochs_train.get_data(copy=False), labels
        except Exception as e:
            raise Exception(e)

    def save_eeg_data(self, X, y, output_path):
        """
        Save EEG data to numpy files.
        """
        try:
            np.savez(output_path, X=X, y=y)
            print(f"Data saved to {output_path}")
        except Exception as e:
            raise Exception(e)