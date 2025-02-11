import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np


class SurfaceDataset(Dataset):
    def __init__(self, file_path, target_offset=1, norm_time=False, split='train', split_ratios=(0.7, 0.2, 0.1)):
        data_raw = xr.open_mfdataset(file_path, combine='by_coords')
        temp = []
        for data_name in data_raw.data_vars:
            if data_name != 'sst' and data_name != 'msl':
                temp.append(data_raw[data_name][:,:128,:128].values)

        data = np.stack(temp, axis=0) # (C, T, H, W)
        data = np.transpose(data, (1, 0, 2, 3)) # (T, C, H, W)
        if norm_time:
            mean = data.mean(axis=0, keepdims=True)  #(1, C, H, W)
            std = data.std(axis=0, keepdims=True)
            data = (data - mean) / std

        self.data = data
        self.target_offset = target_offset
        self.split = split
        T_total = data.shape[0]
        # print(data.shape)

        train_ratio, val_ratio, test_ratio = split_ratios
        if split == 'train':
            start_idx = 0 
            end_idx = int(T_total * train_ratio)
        elif split == 'val':
            start_idx = int(T_total * train_ratio)
            end_idx = int(T_total * (train_ratio + val_ratio))
        elif split == 'test':
            start_idx = int(T_total * (train_ratio + val_ratio))
            end_idx = T_total
        else:
            raise ValueError("split must be 'train', 'val' or 'test'")

        if isinstance(target_offset, int):
            self.target_offsets = [target_offset]
        elif isinstance(target_offset, (list, tuple)):
            self.target_offsets = list(target_offset)
        else:
            raise ValueError("target_offset must be an int or a list/tuple of int")
            
        # 为了保证输入时间 t 和目标 t+offset 都在同一划分内，
        # 有效索引区间为 [start_idx, end_idx - max(target_offsets))
        max_offset = max(self.target_offsets)
        if end_idx - start_idx <= max_offset:
            raise ValueError("Timesteps is not enough for target_offset")
        self.start_idx = start_idx
        self.end_idx = end_idx - max_offset  # 有效样本的最后时间索引: end_idx - max_offset - 1
        self.indices = np.arange(self.start_idx, self.end_idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 根据内部索引获得实际时间 t
        t = self.indices[idx]
        # 输入为时间 t 对应的数据
        x = self.data[t]  # (C, H, W)

        # 支持单个时刻或多个时刻数据
        if len(self.target_offsets) == 1:
            y = self.data[t + self.target_offsets[0]]  #(C, H, W)
        else:
            y = [self.data[t + offset] for offset in self.target_offsets]  # each shape (C, H, W)
            
        x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, list):
            y = [torch.tensor(item, dtype=torch.float32) for item in y]
        else:
            y = torch.tensor(y, dtype=torch.float32)
            
        return x, y

