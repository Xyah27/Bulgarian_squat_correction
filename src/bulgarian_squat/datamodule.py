import torch
from torch.utils.data import Dataset, DataLoader

class RepsDataset(Dataset):
    def __init__(self, reps, labels):
        self.reps = reps        # lista de np.array [T, F]
        self.labels = labels    # lista/np.array [K]
    def __len__(self): return len(self.reps)
    def __getitem__(self, i):
        x = torch.tensor(self.reps[i], dtype=torch.float32)
        y = torch.tensor(self.labels[i], dtype=torch.float32)
        return x, y, x.shape[0]

def collate_pad(batch):
    xs, ys, lens = zip(*batch)
    F = xs[0].shape[1]
    T = max(lens)
    X = torch.zeros(len(xs), T, F)
    M = torch.zeros(len(xs), T, dtype=torch.bool)
    for i,(x,l) in enumerate(zip(xs, lens)):
        X[i,:l] = x
        M[i,:l] = True
    Y = torch.stack(ys)
    return X, Y, M

def make_loader(dataset, bs=32, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, collate_fn=collate_pad, num_workers=num_workers)
