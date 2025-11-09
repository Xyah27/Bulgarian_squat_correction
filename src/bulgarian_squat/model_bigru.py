import torch
import torch.nn as nn

class BiGRUClassifier(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, num_classes=4, dropout=0.3):
        super().__init__()
        self.gru1 = nn.GRU(in_dim, hidden1, batch_first=True, bidirectional=True)
        self.do1  = nn.Dropout(dropout)
        self.gru2 = nn.GRU(2*hidden1, hidden2, batch_first=True, bidirectional=True)
        self.do2  = nn.Dropout(dropout)
        self.fc1  = nn.Linear(2*hidden2, 64)
        self.act  = nn.ReLU()
        self.fc2  = nn.Linear(64, num_classes)  # logits

    def forward(self, x, mask):
        out,_ = self.gru1(x)
        out = self.do1(out)
        out,_ = self.gru2(out)
        out = self.do2(out)
        idx = mask.sum(dim=1) - 1
        last = out[torch.arange(out.size(0)), idx]
        h = self.act(self.fc1(last))
        logits = self.fc2(h)
        return logits
