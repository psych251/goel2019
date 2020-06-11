from typing import List

from torch.utils.data import DataLoader

from replication.ml.data.single_dataset import SingleConditionTouchDataset
from replication.ml.params import BATCH_SIZE, WORKER_NUM
from replication.preprocess.user import User


def collate_data(array):
    return array


def match(a: int, b: int):
    return a == -1 or b == -1 or a == b


TEST_RATIO = 0.3


class UserSplitter:
    user_loaders: List[List[DataLoader]]

    def __init__(self, users: List[User]):
        self.user_loaders = []
        for user in users:
            self.user_loaders += [
                [
                    DataLoader(
                        SingleConditionTouchDataset(user.stressed_condition),
                        batch_size=BATCH_SIZE,
                        collate_fn=collate_data,
                        num_workers=WORKER_NUM
                    ),
                    DataLoader(
                        SingleConditionTouchDataset(user.unstressed_condition),
                        batch_size=BATCH_SIZE,
                        collate_fn=collate_data,
                        num_workers=WORKER_NUM
                    )
                ]
            ]
