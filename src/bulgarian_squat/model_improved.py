# src/model_bilstm_improved.py
"""
Versión mejorada del clasificador BiLSTM con:
- Batch Normalization
- Mejor inicialización
- Layer Normalization opcional
- Attention mechanism opcional
"""
import torch
import torch.nn as nn

class BiLSTMClassifierImproved(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, num_classes=4, 
                 dropout=0.3, use_batch_norm=True, use_attention=False):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm
        
        # Input normalization
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(in_dim)
        
        # First BiLSTM layer
        self.lstm1 = nn.LSTM(in_dim, hidden1, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(2*hidden1)
        self.do1 = nn.Dropout(dropout)
        
        # Second BiLSTM layer
        self.lstm2 = nn.LSTM(2*hidden1, hidden2, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(2*hidden2)
        self.do2 = nn.Dropout(dropout)
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(2*hidden2, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # Fully connected layers
        self.fc1 = nn.Linear(2*hidden2, 64)
        self.bn_fc = nn.BatchNorm1d(64) if use_batch_norm else None
        self.act = nn.ReLU()
        self.do_fc = nn.Dropout(dropout/2)  # Less dropout antes del output
        self.fc2 = nn.Linear(64, num_classes)
        
        # Inicialización Xavier
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización Xavier para mejor convergencia"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'lstm' not in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name and 'lstm' not in name:
                nn.init.zeros_(param)
    
    def forward(self, x, mask):
        # Input normalization (batch norm over features)
        if self.use_batch_norm:
            # x: (B, T, F) -> (B, F, T) -> batch norm -> (B, T, F)
            x = x.permute(0, 2, 1)
            x = self.input_bn(x)
            x = x.permute(0, 2, 1)
        
        # First LSTM
        out, _ = self.lstm1(x)
        out = self.ln1(out)
        out = self.do1(out)
        
        # Second LSTM
        out, _ = self.lstm2(out)
        out = self.ln2(out)
        out = self.do2(out)
        
        # Get last timestep or use attention
        if self.use_attention:
            # Attention pooling
            attn_weights = self.attention(out)  # (B, T, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(attn_weights * out, dim=1)  # (B, 2*hidden2)
        else:
            # Last timestep with mask
            idx = mask.sum(dim=1).long() - 1
            idx = torch.clamp(idx, min=0, max=out.size(1)-1)
            context = out[torch.arange(out.size(0)), idx]
        
        # FC layers
        h = self.fc1(context)
        if self.bn_fc is not None:
            h = self.bn_fc(h)
        h = self.act(h)
        h = self.do_fc(h)
        logits = self.fc2(h)
        
        return logits


class BiGRUClassifierImproved(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, num_classes=4, 
                 dropout=0.3, use_batch_norm=True, use_attention=False):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm
        
        # Input normalization
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(in_dim)
        
        # First BiGRU layer
        self.gru1 = nn.GRU(in_dim, hidden1, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(2*hidden1)
        self.do1 = nn.Dropout(dropout)
        
        # Second BiGRU layer
        self.gru2 = nn.GRU(2*hidden1, hidden2, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(2*hidden2)
        self.do2 = nn.Dropout(dropout)
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(2*hidden2, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # Fully connected layers
        self.fc1 = nn.Linear(2*hidden2, 64)
        self.bn_fc = nn.BatchNorm1d(64) if use_batch_norm else None
        self.act = nn.ReLU()
        self.do_fc = nn.Dropout(dropout/2)
        self.fc2 = nn.Linear(64, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'gru' not in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name and 'gru' not in name:
                nn.init.zeros_(param)
    
    def forward(self, x, mask):
        # Input normalization
        if self.use_batch_norm:
            x = x.permute(0, 2, 1)
            x = self.input_bn(x)
            x = x.permute(0, 2, 1)
        
        # First GRU
        out, _ = self.gru1(x)
        out = self.ln1(out)
        out = self.do1(out)
        
        # Second GRU
        out, _ = self.gru2(out)
        out = self.ln2(out)
        out = self.do2(out)
        
        # Attention or last timestep
        if self.use_attention:
            attn_weights = self.attention(out)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(attn_weights * out, dim=1)
        else:
            idx = mask.sum(dim=1).long() - 1
            idx = torch.clamp(idx, min=0, max=out.size(1)-1)
            context = out[torch.arange(out.size(0)), idx]
        
        # FC layers
        h = self.fc1(context)
        if self.bn_fc is not None:
            h = self.bn_fc(h)
        h = self.act(h)
        h = self.do_fc(h)
        logits = self.fc2(h)
        
        return logits
