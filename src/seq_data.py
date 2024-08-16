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
            "Gen_Phase2_Temp_Avg",
            "Gen_Phase3_Temp_Avg",
            "Gen_SlipRing_Temp_Avg",
        ]
    elif system == "transformer":
        features = [
            "HVTrafo_Phase1_Temp_Avg",
            "HVTrafo_Phase2_Temp_Avg",
            "HVTrafo_Phase3_Temp_Avg",
        ]
    elif system == "Hydraulic":
        features = [
            "Hyd_Oil_Temp_Avg",
            "Amb_Temp_Avg",
            "Amb_WindSpeed_Avg",
            "Amb_WindSpeed_Std",
            "Blds_PitchAngle_Avg",
            "Blds_PitchAngle_Std",
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

def tumbling_window(data, seq_len):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    data_seq = []
    for i in range(0, len(scaled_data) - seq_len + 1, seq_len//2):
        data_seq.append(scaled_data[i : i + seq_len])
    return np.array(data_seq)

def data_process(raw_dir, data_type, case, seq_len, batch_size, system, val_split_ratio):

    features = selected_features(system)

    df = pd.read_csv(f"{raw_dir}\\Wind-Turbine-SCADA-signals-{data_type}.csv")
    turbine_ids = list(df["Turbine_ID"].unique())

    tick = "T06"
    turbine_ids.pop(turbine_ids.index(tick))

    df_turbine = df[df["Turbine_ID"] == tick].copy()

    # Generate histogram distribution for 'Gen_RPM_Avg'
    hist, bin_edges = np.histogram(df_turbine['Gen_RPM_Avg'], bins=100)
    first_bin_max = bin_edges[2]
    # Remove data points that fall into the first bin
    removed_indices = df_turbine[df_turbine['Gen_RPM_Avg'] <= first_bin_max].index

    _df_turbine = df_turbine.drop(removed_indices, errors='ignore')
    _time_stamp = pd.to_datetime(_df_turbine["Timestamp"], format="mixed")
    _df_turbine = _df_turbine[features]

    data = to_sequences(_df_turbine, seq_len)

    if case == "test":
        time_stamp = pd.to_datetime(df_turbine["Timestamp"], format="mixed")
        df_turbine = df_turbine[features]
        # test_data = tumbling_window(df_turbine, seq_len)
        test_data = to_sequences(df_turbine, seq_len)

        test_dataset = ArrayDataset(test_data, None, data_details=tick)
        test_loaders = [DataLoader(test_dataset, batch_size=batch_size, shuffle=False)]

        train_dataset = ArrayDataset(data, None, data_details=tick)
        train_loaders = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True)]

        print(f"Test data seqs: {len(test_data)}")
        return test_loaders, train_loaders, time_stamp, _time_stamp
    
    train_data = data[: int(len(data) * (1 - val_split_ratio))]
    val_data = data[int(len(data) * (1 - val_split_ratio)) :]

    train_dataset = ArrayDataset(train_data, None, data_details=tick)
    train_loaders = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True)]

    val_dataset = ArrayDataset(val_data, None, data_details=tick)
    val_loaders = [DataLoader(val_dataset, batch_size=batch_size, shuffle=False)]

    print(f"Total data seqs: {len(data)} || Train data seqs: {len(train_data)} || Val data seqs: {len(val_data)}")
    return train_loaders, val_loaders, None, _time_stamp

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