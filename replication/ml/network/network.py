from typing import Tuple, Optional, List

import torch
import torch.nn as nn

from replication.ml.tool.padding import pad_input, unpad_output
from .input_network import InputNet
from .lstm_network import LstmNet


class TouchNet(nn.Module):
    def __init__(self, input_dim, middle_dim):
        super(TouchNet, self).__init__()
        self.input_network = InputNet(input_dim, middle_dim)
        self.lstm_network = LstmNet(middle_dim, 2)
        self.linear = nn.Linear(in_features=middle_dim, out_features=2)

    # noinspection PyShadowingBuiltins
    def forward_seq(self, input: List[torch.Tensor]) -> torch.Tensor:
        padded_input, trace_counts, input_lengths = pad_input(input)
        cnn_output = self.input_network(padded_input)
        # noinspection PyUnresolvedReferences
        output_length = self.input_network.process_lengths(input_lengths)
        unpadded_output = unpad_output(cnn_output, trace_counts, output_length)
        summed_output = [output.mean(dim=0).t() for output in unpadded_output]
        # lstm_input = pack_sequence(summed_output, enforce_sorted=False)
        # lstm_output = self.lstm_network(lstm_input)  # Throw away output `hidden`
        # return lstm_output
        linear_input = torch.stack([output.mean(dim=0) for output in summed_output])
        return self.linear(linear_input)

    # noinspection PyShadowingBuiltins
    def forward_single(self, input: torch.Tensor, hidden: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        cnn_output = self.input_network(input)
        lstm_input = cnn_output.transpose(1, 2)
        lstm_output, hidden_output = self.lstm_network(lstm_input, hidden)  # Throw away output `hidden`
        return lstm_output[0], hidden_output

    # noinspection PyShadowingBuiltins
    def forward(self, input, hidden=None):
        if isinstance(input, list):
            return self.forward_seq(input)
        elif isinstance(input, torch.Tensor):
            return self.forward_single(input, hidden)
        else:
            raise ValueError
