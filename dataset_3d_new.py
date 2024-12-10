import datasets
import numpy as np
import xarray as xr
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random

class RWKVWeatherDataset(Dataset):
    def __init__(self, data_path,seq_len=10,max_len=50,output_len=1,flag='train',split=0.8):

        assert seq_len <= max_len//2 
        assert seq_len >= output_len
        assert flag in ['train', 'val']
        assert split >= 0 and split <=1.0
        self.flag = flag
        self.path = data_path
        self.seq_len = seq_len
        self.max_len = max_len
        self.output_len = output_len
        self.split = split
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]
        self.chunks = []
        self.__read_data__()

    def __read_data__(self):

        
        data_raw = xr.open_mfdataset(self.path, combine='by_coords')
        temp = []
        for data_name in data_raw.data_vars:
            if data_name != 'sst':
                temp.append(data_raw[data_name][:,:176,:156].values)

        temp = np.stack(temp, axis=0) # (C, T, H, W)

        time_dim_length = temp.shape[1]

        for start_idx in range(0, time_dim_length, self.max_len):
            end_idx = start_idx + self.max_len
            if end_idx > time_dim_length:
                end_idx = time_dim_length
                break
            chunk = temp[:, start_idx:end_idx, :, :]
            self.chunks.append(chunk) # (C, T, H, W)

        num_train = int(len(self.chunks) * self.split)###
        print(f"Number of {self.flag} samples: {num_train}")
        border1s = [0, num_train]
        border2s = [num_train, len(self.chunks)]
        border1,border2 = border1s[self.set_type],border2s[self.set_type]
        self.chunks = self.chunks[border1:border2]
    
    def __getitem__(self, idx):

        s = random.randrange(0,self.max_len-3*self.seq_len) ###
        input_points = []
        target = []

        for chunk in self.chunks:
            s = random.randrange(0,self.max_len-2*self.seq_len) 
            input_data = chunk[:, s:s+self.seq_len, :, :]
            target_data = chunk[:, s+self.seq_len:s+self.seq_len+self.output_len, :, :]
            input_points.append(input_data)
            target.append(target_data)

        # sample = {
        #     'input': np.concatenate(input_points, axis=1),
        #     'target': np.concatenate(target, axis=1),
        # }

        sample = {
            'input': np.array(input_points)[idx],
            'target': np.array(target)[idx],
        }
        return sample

    def __len__(self):
        print(f"Number of {self.flag} samples: {len(self.chunks)}")
        return len(self.chunks)
        # return len(self.chunks)