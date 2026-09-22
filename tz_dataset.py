'''
this is the dataset class the data loader will use
'''

from torch.utils.data import Dataset
from pathlib import Path

class TZ_Dataset(Dataset):
    def __init__(self, dataset_dir: Path):
        pass

    def __iter__(self):
        pass