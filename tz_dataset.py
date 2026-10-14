'''
this is the dataset class the data loader will use
'''

import torch
from torch.utils.data import Dataset, IterableDataset
from funcmol.models.nffm import sample_normal

# for the validation set
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

# for the infinite training dataset
class NFFMStream(IterableDataset):
    def __init__(self, nffm, width, minibatch, num_samples=256, num_steps=100, eps=1e-4):
        self.nffm, self.width = nffm, width
        self.num_samples, self.num_steps = num_samples, num_steps
        self.eps, self.minibatch = eps, minibatch

    def __iter__(self):
        dt = 1.0 / self.num_steps
        while True:
            x = sample_normal(self.num_samples, self.width).to(self.nffm.device)
            xs, ts = [], []
            
            for i in range(self.num_steps):
                t = i * dt
                t_batch = torch.full((x.shape[0], 1), t, device=x.device)
                xs.append(x); ts.append(t_batch)

                with torch.no_grad():
                    x1_hat = self.nffm(x, t_batch)
                    v = (x1_hat - x) / max(1.0 - t, self.eps)
                    x = x + dt * v
            
            # save the last pair
            xs.append(x); ts.append(t_batch)
            
            X, T = torch.cat(xs), torch.cat(ts)
            perm = torch.randperm(X.shape[0], device=X.device)
            X, T = X[perm], T[perm]
            
            for s in range(0, X.shape[0], self.minibatch):
                yield X[s:s + self.minibatch], T[s:s + self.minibatch]