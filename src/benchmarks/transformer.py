from torch import nn

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_size, n_head, num_layers, output_size):
        super().__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=n_head, dim_feedforward=input_size * 4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = self.linear(output[:, -1, :])
        return self.tanh(output)    