import math
from typing import List

import torch
import torch.nn as nn


class InputNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InputNet, self).__init__()
        self.n = input_dim
        self.kernel1_size = 7
        self.kernel2_size = 5

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=input_dim * 2,
                kernel_size=self.kernel1_size,
                padding=(self.kernel1_size - 1) // 2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim * 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim * 2,
                out_channels=input_dim * 4,
                kernel_size=self.kernel1_size,
                padding=(self.kernel1_size - 1) // 2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel1_size, ceil_mode=True),
            nn.BatchNorm1d(input_dim * 4)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim * 4,
                out_channels=input_dim * 8,
                kernel_size=self.kernel2_size,
                padding=(self.kernel2_size - 1) // 2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim * 8)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim * 8,
                out_channels=input_dim * 16,
                kernel_size=self.kernel2_size,
                padding=(self.kernel2_size - 1) // 2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel2_size, ceil_mode=True),
            nn.BatchNorm1d(input_dim * 16),
        )
        self.dropout = nn.Dropout(p=0.5)
        assert input_dim * 16 == output_dim

    # noinspection PyShadowingBuiltins
    def forward(self, input: torch.Tensor):
        x = input
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        return x

    def process_lengths(self, lengths: List[int]):
        return [math.ceil(math.ceil(length / self.kernel1_size) / self.kernel2_size) for length in lengths]
