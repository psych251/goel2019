from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

from .input_network import InputNet
from .lstm_network import LstmNet


class TouchNet(nn.Module):
    def __init__(self, input_dim, middle_dim):
        super(TouchNet, self).__init__()
        self.input_network = InputNet(input_dim, middle_dim)
        self.lstm_network = LstmNet(middle_dim, 1)

    def forward_seq(self, input: PackedSequence) -> PackedSequence:
        padded_input, input_indexes = pad_packed_sequence(input, batch_first=True)
        cnn_output = [self.input_network(padded_input[[i]].transpose(1, 2))[0].t() for i in range(padded_input.shape[0])]
        lstm_input = torch.nn.utils.rnn.pack_sequence(cnn_output, enforce_sorted=False)
        lstm_output = self.lstm_network(lstm_input)  # Throw away output `hidden`
        return lstm_output

    def forward_single(self, input: torch.Tensor, hidden: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        cnn_output = self.input_network(input)
        lstm_input = cnn_output.view(1, cnn_output.size(0))
        lstm_output, hidden_output = self.lstm_network(lstm_input, hidden)  # Throw away output `hidden`
        return lstm_output[0], hidden_output

    def forward(self, input, hidden=None):
        if isinstance(input, PackedSequence):
            return self.forward_seq(input)
        elif isinstance(input, torch.Tensor):
            return self.forward_single(input, hidden)
        else:
            raise ValueError
