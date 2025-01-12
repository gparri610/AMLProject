from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        out = self.linear(x[:, -1, :])
        return self.tanh(out)

