import torch
from torch.utils.data import Dataset
from lib.utils import get_slice_params

class ElevatorSlices(Dataset):
    def __init__(self, slices):
        self.slices = slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        sample_slice = self.slices[idx]
        sample_vec = sample_slice.reshape(-1, order='F')
        data = torch.from_numpy(sample_vec).float()
        return data, data
    
class ElevatorSlicesChannels(Dataset):
    def __init__(self, slices):
        self.slices = slices

    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        sample_slice = self.slices[idx]
        data = torch.tensor(sample_slice, dtype=torch.float32).T
        return data, data
    
class ElevatorSliceParams(Dataset):
    def __init__(self, slices):
        self.slices = slices

    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        sample_slice = self.slices[idx]
        params = get_slice_params(sample_slice)
        data = torch.tensor(params, dtype=torch.float32)
        return data, data