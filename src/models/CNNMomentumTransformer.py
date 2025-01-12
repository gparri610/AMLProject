import torch
from torch import nn
from models import LSTMEncoder, Conv1DNet, GLU_custom, InterpretableMHA

class CNNMomentumTransformer(nn.Module):
    """
    This is a variation of the main model developed in our reference paper.
    The GRN components are replaced by CNN networks. The different time steps of the sequences passed are treated as channels.
    The first CNN component takes a sequence as input and also gives a sequence as output.
    The second CNN component takes only one time step (channel) as input and gives one time step (channel) as output.
    Instead of taking the last time step after the Attention component we could also use the CNN layer to size down. This could easily be implemented with the given architecture.

    The forward method has been implemented according to exhibit 14
    """
    def __init__(self, d_model: int, hidden_size: int, dropout_rate: float = 0.3, n_output: int = 1, n_head: int = 5, sequence_length: int = 20, num_filters: int = 32, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__d_model = d_model
        self.__hidden_size = hidden_size
        self.__dropout_rate = dropout_rate
        self.__sequence_length = sequence_length
        self.__num_filters = num_filters

        self.LSTM_encoder = LSTMEncoder(input_size=self.__d_model, hidden_size=self.__hidden_size, dropout_rate=self.__dropout_rate)

        self.glu_1 = GLU_custom(self.__d_model, self.__dropout_rate)
        self.norm_1 = nn.LayerNorm(self.__d_model)

        self.cnn1d_1 = Conv1DNet(input_channels=self.__sequence_length, num_filters=self.__num_filters, kernel_size=7, series_output=True)

        self.att =  InterpretableMHA(self.__d_model, n_head)
        
        self.glu_2 = GLU_custom(self.__d_model, self.__dropout_rate)
        self.norm_2 = nn.LayerNorm(self.__d_model)

        self.cnn1d_2 = Conv1DNet(input_channels=1, num_filters=self.__num_filters, kernel_size=7, series_output=False)

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
        x_2 = self.LSTM_encoder(x)

        x_3 = self.glu_1(x_2)

        x_4 = self.norm_1(x + x_3) # Normalized Residual Connection 1

        x_4_last = x_4[:,-1,:] # Storing the last time step values for the Residual Connection 2

        x_5 = self.cnn1d_1(x_4)
        x_5_last = x_5[:,-1,:] # Storing the last time step values for the Residual Connection 3

        # multihead self-attention         
        x_6, attention_weights = self.att(x_5, x_5, x_5)
        x_6_last = x_6[:,-1,:] # Proceeding with the last time step of each sequence

        x_7 = self.glu_2(x_6_last)

        x_8 = self.norm_2(x_5_last + x_7) # Normalized Residual Connection 3
        x_8 = x_8.unsqueeze(dim=1) # Usnqueezing so that the channel size is 1

        x_9 = self.cnn1d_2(x_8)
        x_9 = x_9.squeeze(dim=1) # Squeezing so we get rid of the channel dimension

        x_10 = self.glu_3(x_9)

        x_11 = self.norm_3(x_4_last + x_10) # Normalized Residual Connection 2 

        # DMN Framework (Dense & Tanh Block) 
        x_12 = self.fc(x_11)

        x_13 = self.tanh(x_12)

        return x_13