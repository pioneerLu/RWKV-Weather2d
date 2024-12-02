import datasets
import numpy as np
import xarray as xr
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class GeoDataset(Dataset):
    def __init__(self, args, data_path, seq_len, input_len, output_len,do_normalize=False,
                split=0.8, flag='train', lat=1, lon=0):
        self.args = args
        self.path = data_path
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.data_list = []
        self.n_window_list = []
        self.all_data = None
        self.do_normalize = do_normalize
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        assert flag in ['train', 'val', 'test']
        assert split >= 0 and split <=1.0
        self.__read_data__(lat, lon)

    def __read_data__(self, lat=1, lon=0):

        assert lat >= 0 and lat < 32
        assert lon >= 0 and lon < 64

        for nc_file in os.listdir(self.path):
            ds_temp = xr.open_mfdataset(os.path.join(self.path, nc_file), combine='by_coords')
            z_temp = ds_temp['z'].isel(lat=lat, lon=lon).to_numpy()
            self.data_list.append(z_temp)

        self.all_data = np.concatenate(self.data_list, axis=0).reshape(-1, 1)

        num_train = int(len(self.all_data) * self.split)
        border1s = [0, num_train - self.seq_len]
        border2s = [num_train, len(self.all_data)]

        border1,border2 = border1s[self.set_type],border2s[self.set_type]
        
        if self.do_normalize:## some question here
            train_data = self.all_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            self.all_data = self.scaler.transform(self.all_data)


    def __getitem__(self, index) -> dict:

        shifted_list = []
        for i in range(1, 5):
            # NumPy 切片实现 shift
            shifted_data = np.zeros_like(self.all_data) 
            shifted_data[i:] = self.all_data[:-i]       
            shifted_list.append(shifted_data[:, np.newaxis])

        seq_x = np.concatenate(shifted_list, axis=1)
        seq_y = self.all_data.reshape(-1,1)
        return dict(targets=seq_y, shifted_targets=seq_x)


    def __len__(self) -> int:

        return len(self.all_data) - self.input_len - self.output_len + 1
