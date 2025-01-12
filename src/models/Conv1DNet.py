import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DNet(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, series_output=True):
        super(Conv1DNet, self).__init__()
        if series_output:
            output_channels = input_channels
        else:
            output_channels = 1
            
        # Ensure the kernel_size is an odd number to facilitate padding calculation
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size should be odd to retain the same dimension.")

        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(num_filters, output_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x
