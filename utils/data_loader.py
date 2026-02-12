import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def load_npz_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y)
    
    return X_tensor, y_tensor

def get_dataset_list(data_dir="datasets"):
    return [f for f in os.listdir(data_dir) if f.endswith('.npz')]