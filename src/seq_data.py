import random
import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset, DataLoader

from utils import train

preprocessing_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])

# Features to be used for training
def selected_features(system):
    if system == "gearbox":
        features = [
            "Gear_Bear_Temp_Avg", 
            "Gear_Oil_Temp_Avg",
            # "Amb_WindSpeed_Avg",
            # "Amb_WindSpeed_Std",
            # "Gen_RPM_Avg",
            # "Gen_RPM_Std",
            # "Blds_PitchAngle_Avg",
            # "Blds_PitchAngle_Std",
        ]
    elif system == "generator":
        features = [
            "Gen_Bear_Temp_Avg",
            "Gen_RPM_Avg",
            "Gen_RPM_Std",
            "Gen_Phase1_Temp_Avg",
            # "Gen_Phase2_Temp_Avg",
            # "Gen_Phase3_Temp_Avg",
            # "Gen_SlipRing_Temp_Avg",
        ]
    elif system == "transformer":
        features = [
            "HVTrafo_Phase1_Temp_Avg",
            "HVTrafo_Phase2_Temp_Avg",
            "HVTrafo_Phase3_Temp_Avg",
        ]
    elif system == "hydraulic":
        features = [
            "Hyd_Oil_Temp_Avg",
            "Amb_Temp_Avg",
            "Amb_WindSpeed_Avg",
            "Amb_WindSpeed_Std",
            # "Blds_PitchAngle_Avg",
            # "Blds_PitchAngle_Std",
        ]

    elif system == "nacelle":
        features = [
            "Nac_Temp_Avg",
            "Rtr_RPM_Avg",
        ]

    return features

# Helper function to set seeds for reproducibility
def set_seeds(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# function to transform data into sequences
def to_sequences(data, seq_len):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    data_seq = []
    for i in range(len(scaled_data) - seq_len + 1):
        data_seq.append(scaled_data[i : i + seq_len])
    return np.array(data_seq)

def tumbling_window(data, seq_len, step_size):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    data_seq = []
    for i in range(0, len(scaled_data) - seq_len + 1, step_size):
        data_seq.append(scaled_data[i : i + seq_len])
    return np.array(data_seq)


def load_data(raw_dir, data_type, features):

    df = pd.read_csv(f"{raw_dir}\\Wind-Turbine-SCADA-signals-{data_type}.csv")
    turbine_ids = list(df["Turbine_ID"].unique())

    tick = "T06"
    turbine_ids.pop(turbine_ids.index(tick))

    unmasked_df = df[df["Turbine_ID"] == tick].copy()

    # Generate histogram distribution for 'Gen_RPM_Avg'
    hist, bin_edges = np.histogram(unmasked_df['Gen_RPM_Avg'], bins=100)
    first_bin_max = bin_edges[2]
    # Remove data points that fall into the first bin
    removed_indices = unmasked_df[unmasked_df['Gen_RPM_Avg'] <= first_bin_max].index

    masked_df = unmasked_df.drop(removed_indices, errors='ignore')
    masked_time = pd.to_datetime(masked_df["Timestamp"], format="mixed")
    masked_df = masked_df[features]

    return masked_df, unmasked_df, masked_time, tick



def data_process(raw_dir, data_type, case, seq_len, batch_size, system, val_split_ratio):

    features = selected_features(system)

    masked_df, unmasked_df, masked_time, tick = load_data(raw_dir, data_type, features)
    
    masked_data = to_sequences(masked_df, seq_len)
    
    if case == "test":
        unmasked_time = pd.to_datetime(unmasked_df["Timestamp"], format="mixed")
        unmasked_df = unmasked_df[features]
        unmasked_data = tumbling_window(unmasked_df, seq_len, seq_len)
        # test_data = to_sequences(df_turbine, seq_len)

        test_dataset = ArrayDataset(unmasked_data, None, data_details=tick)
        test_loaders = [DataLoader(test_dataset, batch_size=batch_size, shuffle=False)]

        train_dataset = ArrayDataset(masked_data, None, data_details=tick)
        train_loaders = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True)]

        print(f"Test data seqs: {len(unmasked_data)}")
        return test_loaders, train_loaders, unmasked_time, masked_time
    

    train_data = masked_data[: int(len(masked_data) * (1 - val_split_ratio))]
    val_data = masked_data[int(len(masked_data) * (1 - val_split_ratio)) :]

    train_dataset = ArrayDataset(train_data, None, data_details=tick)
    train_loaders = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True)]

    val_dataset = ArrayDataset(val_data, None, data_details=tick)
    val_loaders = [DataLoader(val_dataset, batch_size=batch_size, shuffle=False)]

    print(f"Total data seqs: {len(masked_data)} || Train data seqs: {len(train_data)} || Val data seqs: {len(val_data)}")
    return train_loaders, val_loaders, None, masked_time


class ArrayDataset(Dataset):
    all_dataset_names = []
    total_samples = 0

    def __init__(self, data, target, data_details=None):
        self.data_details = data_details
        self.data = torch.from_numpy(data).float()
        self.target = None if target is None else torch.from_numpy(target).float()

        # Save data details and accumulate dataset names
        if data_details is not None:
            ArrayDataset.all_dataset_names.append(
                data_details
            )  # Accumulate dataset name
        ArrayDataset.total_samples += len(
            self
        )  # Accumulate the total number of samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.target is None:
            return self.data[idx]
        else:
            return self.data[idx], self.target[idx]

    @classmethod
    def get_total_samples(cls):
        return cls.total_samples

    @classmethod
    def get_all_dataset_names(cls):
        return cls.all_dataset_names