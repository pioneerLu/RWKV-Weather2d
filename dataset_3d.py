import datasets
import numpy as np
import xarray as xr
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random

class RWKVWeatherDataset(Dataset):
    def __init__(self, data_path,seq_len=10,max_len=50,output_len=10):
        assert seq_len <= max_len//2 
        self.path = data_path
        self.seq_len = seq_len
        self.max_len = max_len
        self.output_len = output_len
        self.chunks = []
        self.__read_data__()

    def __read_data__(self):
        data_raw = xr.open_mfdataset(self.path, combine='by_coords')
        temp = []
        for data_name in data_raw.data_vars:
            temp.append(data_raw[data_name].values)

        temp = np.stack(temp, axis=0) # (C, T, H, W)

        time_dim_length = temp.shape[1]

        for start_idx in range(0, time_dim_length, self.max_len):
            end_idx = start_idx + self.max_len
            if end_idx > time_dim_length:
                end_idx = time_dim_length
            chunk = temp[:, start_idx:end_idx, :, :]
            self.chunks.append(chunk) # (C, T, H, W)


    
    def __getitem__(self, idx):

        """
            这里暂时返回整个对应索引的数据，后续需要完善。
        """
        s = random.randrange(0,self.max_len-2*self.seq_len) ###
        input_points = []
        target = []
        for num_chunk in range(len(self.chunks)):
            s = random.randrange(0,self.max_len-2*self.seq_len) 
            input_data = self.chunks[:, s:s+self.seq_len, :, :]
            target_data = self.chunks[:, s+self.seq_len:s+self.seq_len+self.output_len, :, :]
            input_points.append(input_data)
            target.append(target_data)

        sample = {
            'input': np.concatenate(input_points, axis=1),
            'target': np.concatenate(target, axis=1),
        }

        return sample

    def __len__(self):

        return self.seq_len*len(self.chunks)
        # return len(self.chunks)