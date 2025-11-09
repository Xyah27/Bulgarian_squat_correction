# src/model_bilstm.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMClassifier(nn.Module):
    """
    Bi-LSTM con 2 capas + MLP final.
    Acepta forward(x, mask) o forward(x, lengths) o forward(x).
    - x: (B, T, F)
    - mask: (B, T) bool/int donde True/1 = timesteps válidos (para longitudes)
    - lengths: (B,) long con longitudes por secuencia
    """
    def __init__(self, in_dim, hidden1=128, hidden2=64, num_classes=4, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, hidden1, batch_first=True, bidirectional=True)
        self.do1   = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(2*hidden1, hidden2, batch_first=True, bidirectional=True)
        self.do2   = nn.Dropout(dropout)
        self.fc1   = nn.Linear(2*hidden2, 64)
        self.act   = nn.ReLU()
        self.fc2   = nn.Linear(64, num_classes)  # logits

    def _select_last_valid(self, seq_out, lengths):
        """
        seq_out: (B, T, H), lengths: (B,) long
        Devuelve último estado válido por secuencia.
        """
        B, T, H = seq_out.size()
        idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, H)  # (B,1,H)
        last = seq_out.gather(dim=1, index=idx).squeeze(1)              # (B,H)
        return last

    def forward(self, x, mask=None, lengths=None):
        # x: (B,T,F)
        if lengths is None and mask is not None:
            lengths = mask.long().sum(dim=1)  # (B,)
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)

        # LSTM 1: pack -> LSTM -> unpack -> dropout
        packed1 = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out1, _ = self.lstm1(packed1)
        out1_pad, _ = pad_packed_sequence(out1, batch_first=True)  # (B,T,H1*2)
        out1_pad = self.do1(out1_pad)

        # LSTM 2: repack -> LSTM -> unpack -> dropout
        packed2 = pack_padded_sequence(out1_pad, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out2, _ = self.lstm2(packed2)
        out2_pad, _ = pad_packed_sequence(out2, batch_first=True)  # (B,T,H2*2)
        out2_pad = self.do2(out2_pad)

        # Selección del último timestep válido según lengths
        last = self._select_last_valid(out2_pad, lengths.to(out2_pad.device))
        h = self.act(self.fc1(last))
        logits = self.fc2(h)
        return logits
