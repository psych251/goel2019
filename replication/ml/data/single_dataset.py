import torch
import torch.utils.data

from replication.ml.data.dataset import TouchDataset
from replication.preprocess.condition import Condition


class SingleConditionTouchDataset(torch.utils.data.Dataset):
    condition: Condition

    def __init__(self, condition: Condition):
        self.condition = condition

    # noinspection PyArgumentList
    def __getitem__(self, index):
        return torch.Tensor(
            TouchDataset.separated_track_pad_to_list(self.condition.tasks[index].separated_track_pad_entries)
        )

    def __len__(self):
        return len(self.condition.tasks)
