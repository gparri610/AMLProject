import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=input_size, batch_first=True)

    def forward(self, x):
        x_1, _ = self.lstm1(x)
        x_2 = self.dropout(x_1)
        out, _ = self.lstm2(x_2)
        return out