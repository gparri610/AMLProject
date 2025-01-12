import torch
from torch import nn
from models import GRUEncoder, GRN, GLU_custom

class GRUMomentumTransformer(nn.Module):
    """
    This is a slight variation of the model from the reference paper.
    Instead of an LSTM encoder a GRU encoder is used.
    """
    def __init__(self, d_model: int, hidden_size: int, dropout_rate: float = 0.3, n_output: int = 1, n_head: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__d_model = d_model
        self.__hidden_size = hidden_size
        self.__dropout_rate = dropout_rate

        self.GRU_encoder = GRUEncoder(input_size=self.__d_model, hidden_size=self.__hidden_size, dropout_rate=self.__dropout_rate)

        self.glu_1 = GLU_custom(self.__d_model, self.__dropout_rate)
        self.norm_1 = nn.LayerNorm(self.__d_model)

        self.grn_1 = GRN(self.__d_model, self.__dropout_rate)

        self.att = nn.MultiheadAttention(self.__d_model, n_head, dropout=self.__dropout_rate)
        
        self.glu_2 = GLU_custom(self.__d_model, self.__dropout_rate)
        self.norm_2 = nn.LayerNorm(self.__d_model)

        self.grn_2 = GRN(self.__d_model, self.__dropout_rate)

        self.glu_3 = GLU_custom(self.__d_model, self.__dropout_rate)
        self.norm_3 = nn.LayerNorm(self.__d_model)

        # DMN Framework (Dense & Tanh Block) 
        self.fc = nn.Linear(self.__d_model, n_output)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Data Format: B x T x D
        B: Batch Size
        T: Time Steps
        D: Features (for each time step)

        """
        x_2 = self.GRUEncoder(x)

        x_3 = self.glu_1(x_2)

        x_4 = self.norm_1(x + x_3) # Normalized Residual Connection 1

        x_4_last = x_4[:,-1,:] # Storing the last time step values for the Residual Connection 2

        x_5 = self.grn_1(x_4)
        x_5_last = x_5[:,-1,:] # Storing the last time step values for the Residual Connection 3

        # Multihead Self-Attention         
        x_6, attention_weights = self.att(x_5, x_5, x_5)
        x_6_last = x_6[:,-1,:] # Proceeding with the last time step of each sequence

        x_7 = self.glu_2(x_6_last)

        x_8 = self.norm_2(x_5_last + x_7) # Normalized Residual Connection 3

        x_9 = self.grn_2(x_8)

        x_10 = self.glu_3(x_9)

        x_11 = self.norm_3(x_4_last + x_10) # Normalized Residual Connection 2 

        # DMN Framework (Dense & Tanh Block) 
        x_12 = self.fc(x_11)

        x_13 = self.tanh(x_12)

        return x_13
