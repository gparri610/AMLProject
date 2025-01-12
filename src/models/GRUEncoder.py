import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=input_size, batch_first=True)

    def forward(self, x):
        x_1, _ = self.gru1(x)
        x_2 = self.dropout(x_1)
        out, _ = self.gru2(x_2)
        return out