import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU_custom(nn.Module):
    def __init__(self, d_model, dropout_rate):
        super().__init__()
        # Define weights and biases for the GLU
        self.W4 = nn.Parameter(torch.randn(d_model, d_model))
        self.b4 = nn.Parameter(torch.randn(d_model))
        self.W5 = nn.Parameter(torch.randn(d_model, d_model))
        self.b5 = nn.Parameter(torch.randn(d_model))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, gamma):
        # Compute the GLU operation
        x = F.linear(gamma, self.W4, self.b4)
        y = F.linear(gamma, self.W5, self.b5)
        x = torch.sigmoid(x) * y  # Element-wise multiplication (Hadamard product)
        return x