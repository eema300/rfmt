'''
this is the dataset class the data loader will use
'''

from pathlib import Path
import torch
from torch.utils.data import Dataset


class TZ_Dataset(Dataset):
    def __init__(self, x_t: torch.Tensor, t: torch.Tensor):
        self.x_t = x_t
        self.t = t

    def __len__(self):
        return self.x_t.shape[0]

    def __getitem__(self, key):
        return self.x_t[key], self.t[key]

    def add_data(self, x_t: torch.Tensor, t: torch.Tensor):
        self.x_t = torch.cat((self.x_t, x_t.detach().cpu()), dim=0)
        self.t = torch.cat((self.t, t.detach().cpu()), dim=0)