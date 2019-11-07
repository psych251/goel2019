from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import torch.nn.functional as f


class LstmNet(nn.Module):
    def __init__(self, input_num, out_num):
        super(LstmNet, self).__init__()

        self.lstm = nn.LSTM(input_size=input_num, hidden_size=input_num, num_layers=2, batch_first=True)
        self.mlp7 = nn.Linear(in_features=input_num, out_features=out_num)
        self.mlp7_bn = nn.BatchNorm1d(out_num)

    def forward_seq(self, lstm_input: PackedSequence) -> PackedSequence:
        lstm_output, _ = self.lstm(lstm_input)  # Throw away output `hidden`
        output, output_sizes = pad_packed_sequence(lstm_output, batch_first=True)
        last_seq_idxs = torch.LongTensor([x - 1 for x in output_sizes])
        last_seq_items = output[range(output.shape[0]), last_seq_idxs, :]
        x = self.mlp7(last_seq_items)
        x = self.mlp7_bn(x)
        return x

    def forward_single(self, lstm_input: torch.Tensor, hidden: Optional[torch.Tensor])\
            -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_output, hidden_output = self.lstm(lstm_input, hidden)
        x = self.mlp7(lstm_output[0, -1])
        # x = self.mlp7_bn(x)
        return x, hidden_output

    def forward(self, input, hidden=None):
        if isinstance(input, PackedSequence):
            return self.forward_seq(input)
        elif isinstance(input, torch.Tensor):
            return self.forward_single(input, hidden)
