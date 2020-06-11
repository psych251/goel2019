from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from replication.ml.tool.padding import pad_input, unpad_output, unpad_output_by_trace, unpad_output_traces
from .input_network import InputNet
from .lstm_network import LstmNet


class TouchNet(nn.Module):
    def __init__(self, input_dim, middle_dim):
        super(TouchNet, self).__init__()
        self.input_network = InputNet(input_dim, middle_dim)
        self.lstm_network = LstmNet(middle_dim, 2)
        self.linear = nn.Sequential(
            nn.Linear(in_features=middle_dim // 2, out_features=middle_dim // 4),
            nn.ReLU(),
            nn.Linear(in_features=middle_dim // 4, out_features=2),
        )

    # noinspection PyShadowingBuiltins
    def forward_seq(self, input: List[torch.Tensor]) -> torch.Tensor:
        padded_input, trace_counts, input_lengths = pad_input(input)
        cnn_output = self.input_network(padded_input)
        # noinspection PyUnresolvedReferences
        output_length = self.input_network.process_lengths(input_lengths)
        unpadded_output = unpad_output(cnn_output, trace_counts, output_length)
        # summed_output = [
        #     (output[:, :, range(output.shape[2] / 2)] *
        #     output[:, :, output.shape[2] / 2 + range(output.shape[2] / 2)]).sum(dim=2) /
        #     output[:, :, range(output.shape[2] / 2)].sum(dim=2)
        #     for output in unpadded_output
        # ]
        summed_output = []
        for output in unpadded_output:
            weight = torch.sigmoid(output[:, range(output.shape[1] // 2), :])
            value = output[:, range(output.shape[1] // 2, output.shape[1]), :]
            weight_sum = weight.sum(dim=2) + 0.001
            weighted_sum = (value * weight).sum(dim=2)
            weighted_mean = weighted_sum / weight_sum
            summed_output += [weighted_mean]
        linear_input = torch.cat(summed_output)
        linear_output = self.linear(linear_input)
        unpadded_linear_output = unpad_output_traces(linear_output, trace_counts)
        summed_linear_output = []
        for output in unpadded_linear_output:
            weight = torch.sigmoid(output[:, 0])
            value = output[:, 1]
            weight_sum = weight.sum() + 0.001
            weighted_sum = (value * weight).sum()
            weighted_mean = weighted_sum / weight_sum
            summed_linear_output += [weighted_mean]
        final_output = torch.stack(summed_linear_output)
        return final_output
        # unpadded_output_traces = unpad_output_by_trace(cnn_output, trace_counts, output_length)
        # lstm_input = [input.t() for input in unpadded_output_traces]
        # packed_lstm_input = pack_sequence(lstm_input, enforce_sorted=False)
        # lstm_output = self.lstm_network(packed_lstm_input)
        # grouped_lstm_output = unpad_output_traces(lstm_output, trace_counts)
        # assert (grouped_lstm_output[0].shape[1] == 2)
        # summed_output = [(output[:, 0] * output[:, 1]).sum() / (output[:, 1].sum() + 0.01) for output in
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
        # summed_output = [
        #     (output[:, :, range(output.shape[2] / 2)] *
        #     output[:, :, output.shape[2] / 2 + range(output.shape[2] / 2)]).sum(dim=2) /
        #     output[:, :, range(output.shape[2] / 2)].sum(dim=2)
        #     for output in unpadded_output
        # ]
        summed_output = []
        for output in unpadded_output:
            weight = torch.sigmoid(output[:, range(output.shape[1] // 2), :])
            value = output[:, range(output.shape[1] // 2, output.shape[1]), :]
            weight_sum = weight.sum(dim=2) + 0.001
            weighted_sum = (value * weight).sum(dim=2)
            weighted_mean = weighted_sum / weight_sum
            summed_output += [weighted_mean]
        # linear_input = torch.cat(summed_output)
        # linear_output = self.linear(linear_input)
        # unpadded_linear_output = unpad_output_traces(linear_output, trace_counts)
        # unpadded_summed_output = unpad_output_traces(linear_input, trace_counts)
        # summed_linear_output = []
        # for linear, output in zip(unpadded_linear_output, unpadded_summed_output):
        #     weight = torch.sigmoid(linear[:, 1])
        #     value = output
        #     weight_sum = weight.sum(dim=0) + 0.001
        #     weighted_sum = (value.T * weight).T.sum(dim=0)
        #     weighted_mean = weighted_sum / weight_sum
        #     summed_linear_output += [weighted_mean]
        # final_output = torch.stack(summed_linear_output)
        return summed_output