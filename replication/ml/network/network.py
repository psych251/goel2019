from typing import Tuple, Optional, List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from replication.ml.tool.padding import pad_input, unpad_output, unpad_output_by_trace, unpad_output_traces
from .input_network import InputNet
from .lstm_network import LstmNet


class TouchNet(nn.Module):
    def __init__(self, input_dim, middle_dim):
        super(TouchNet, self).__init__()
        self.input_network = InputNet(input_dim, middle_dim)
        self.lstm_network = LstmNet(middle_dim, 2)
        self.linear = nn.Linear(in_features=middle_dim, out_features=1)

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
        return self.linear(linear_input).mean(1)
        # unpadded_output_traces = unpad_output_by_trace(cnn_output, trace_counts, output_length)
        # lstm_input = [input.t() for input in unpadded_output_traces]
        # packed_lstm_input = pack_sequence(lstm_input, enforce_sorted=False)
        # lstm_output = self.lstm_network(packed_lstm_input)
        # grouped_lstm_output = unpad_output_traces(lstm_output, trace_counts)
        # assert (grouped_lstm_output[0].shape[1] == 2)
        # summed_output = [(output[:, 0] * output[:, 1]).sum() / (output[:, 1].sum() + 1e-7) for output in
        #                  grouped_lstm_output]
        # output = torch.stack(summed_output)
        # return output

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

    def forward_features(self, input):
        padded_input, trace_counts, input_lengths = pad_input(input)
        cnn_output = self.input_network(padded_input)
        # noinspection PyUnresolvedReferences
        output_length = self.input_network.process_lengths(input_lengths)
        unpadded_output = unpad_output(cnn_output, trace_counts, output_length)
        summed_output = [output.mean(dim=2) for output in unpadded_output]
        return summed_output