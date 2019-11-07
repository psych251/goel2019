import torch.nn as nn
import torch.nn.functional as f
import torch


class InputNet(nn.Module):
    def __init__(self, input_dim, middle_dim):
        super(InputNet, self).__init__()
        self.n = input_dim
        self.kernel1_size = 7
        self.kernel2_size = 5

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim * 2,
            kernel_size=self.kernel1_size,
            padding=self.kernel1_size
        )
        self.conv1_bn = nn.BatchNorm1d(input_dim * 2)

        self.conv2 = nn.Conv1d(
            in_channels=input_dim * 2,
            out_channels=input_dim * 4,
            kernel_size=self.kernel1_size,
            padding=self.kernel1_size
        )
        self.maxpool2 = nn.MaxPool1d(kernel_size=self.kernel1_size)
        self.conv2_bn = nn.BatchNorm1d(input_dim * 4)

        self.conv3 = nn.Conv1d(
            in_channels=input_dim * 4,
            out_channels=input_dim * 8,
            kernel_size=self.kernel2_size,
            padding=self.kernel2_size
        )
        self.conv3_bn = nn.BatchNorm1d(input_dim * 8)

        self.conv4 = nn.Conv1d(
            in_channels=input_dim * 8,
            out_channels=input_dim * 16,
            kernel_size=self.kernel2_size,
            padding=self.kernel2_size
        )
        self.maxpool4 = nn.MaxPool1d(kernel_size=self.kernel1_size)
        self.conv4_bn = nn.BatchNorm1d(input_dim * 16)
        self.dropout = nn.Dropout(p=0.5)
        assert input_dim * 16 == middle_dim

    # noinspection PyShadowingBuiltins
    def forward(self, input: torch.Tensor):
        x = input
        x = f.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = f.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.maxpool2(x)
        x = f.relu(self.conv3(x))
        x = self.conv3_bn(x)
        x = f.relu(self.conv4(x))
        x = self.conv4_bn(x)
        x = self.maxpool4(x)
        return x
