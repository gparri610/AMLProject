import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GLU_custom


class GRN(nn.Module):
    """
    This implementation is based on the paper: "Temporal Fusion Transformers
    for Interpretable Multi-horizon Time Series Forecasting". The network is described by equation 2 - 4.
    """
    def __init__(self, d_model, dropout_rate) -> None:
        super().__init__()
        # Define the dimensions for the model
        self.d_model = d_model
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # GLU_custom as defined in the given task
        self.glu = GLU_custom(self.d_model, dropout_rate)
        
        # Weights and biases for linear transformations
        self.W1 = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.b1 = nn.Parameter(torch.randn(self.d_model))
        self.W2 = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.W3 = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.b2 = nn.Parameter(torch.randn(self.d_model))

    def forward(self, a, c=None):
        # Initial input context vector c can be optional
        if c is None:
            c = torch.zeros_like(a)

        # Compute Eq. 4: eta2 = ELU(W2*a + W3*c + b2)
        eta2 = F.elu(F.linear(a, self.W2) + F.linear(c, self.W3) + self.b2)

        # Compute Eq. 3: eta1 = W1*eta2 + b1
        eta1 = F.linear(eta2, self.W1) + self.b1

        # Compute Eq. 2: LayerNorm(a + GLU(eta1))
        norm_input = a + self.glu(eta1)
        output = self.layer_norm(norm_input)
        
        return output
