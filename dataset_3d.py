import datasets
import numpy as np
import xarray as xr
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random

class RWKVWeatherDataset(Dataset):
    def __init__(self, data_path,seq_len=10,input_len=10,output_len=1):
        self.path = data_path
        self.seq_len = seq_len
        self.chunks = []
        self.__read_data__()

    def __read_data__(self):
        data_raw = xr.open_mfdataset(self.path, combine='by_coords')
        temp = []
        for data_name in data_raw.data_vars:
            temp.append(data_raw[data_name].values)

        temp = np.stack(temp, axis=0) # (C, T, H, W)

        time_dim_length = temp.shape[1]

        for start_idx in range(0, time_dim_length, self.seq_len):
            end_idx = start_idx + self.seq_len
            if end_idx > time_dim_length:
                end_idx = time_dim_length
            chunk = temp[:, start_idx:end_idx, :, :]
            self.chunks.append(chunk)


    
    def __getitem__(self, idx):

        """
            这里暂时返回整个对应索引的数据，后续需要完善。
        """
        s = random.randrange(0,self.seq_len)###

        input_data = self.chunks[:, idx:idx + self.input_len, :, :]
        target_data = self.chunks[:, idx + self.input_len:idx + self.input_len + self.output_len, :, :]

        sample = {
            'input': input_data,
            'target': target_data
        }

        return sample

    def __len__(self):

        return self.all_data.shape[1] - self.input_len