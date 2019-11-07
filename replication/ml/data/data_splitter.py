import random

from torch.utils.data import DataLoader
from typing import List

from replication.ml.data.dataset import TouchDataset
from replication.ml.params import BATCH_SIZE, WORKER_NUM
from replication.preprocess.user import User


def collate_data(array):
    data = [[], []]
    for x, y in array:
        data[0] += [x]
        data[1] += [y]
    return data


def match(a: int, b: int):
    return a == -1 or b == -1 or a == b


VAL_RATIO = 0.1
TEST_RATIO = 0.1
TRAIN_RATIO = 1 - VAL_RATIO - TEST_RATIO


class DataSplitter:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    def __init__(self, users: List[User]):
        train_users = []
        val_users = []
        test_users = []

        for user in users:
            train_user = User()
            val_user = User()
            test_user = User()

            stressed_tasks = user.stressed_condition.tasks
            random.shuffle(stressed_tasks)
            stressed_train = len(stressed_tasks) * TRAIN_RATIO
            train_threshold = int(stressed_train)
            stressed_val = len(stressed_tasks) * VAL_RATIO
            val_threshold = int(stressed_train + stressed_val)
            stressed_test = len(stressed_tasks) * TEST_RATIO
            test_threshold = int(stressed_train + stressed_val + stressed_test)
            train_user.stressed_condition.tasks = \
                stressed_tasks[0: train_threshold]
            val_user.stressed_condition.tasks = \
                stressed_tasks[train_threshold: val_threshold]
            test_user.stressed_condition.tasks = \
                stressed_tasks[val_threshold: test_threshold]

            unstressed_tasks = user.unstressed_condition.tasks
            random.shuffle(unstressed_tasks)
            unstressed_train = len(unstressed_tasks) * TRAIN_RATIO
            train_threshold = int(unstressed_train)
            unstressed_val = len(unstressed_tasks) * VAL_RATIO
            val_threshold = int(unstressed_train + unstressed_val)
            unstressed_test = len(unstressed_tasks) * TEST_RATIO
            test_threshold = int(unstressed_train + unstressed_val + unstressed_test)
            train_user.unstressed_condition.tasks = \
                unstressed_tasks[0: train_threshold]
            val_user.unstressed_condition.tasks = \
                unstressed_tasks[train_threshold: val_threshold]
            test_user.unstressed_condition.tasks = \
                unstressed_tasks[val_threshold: test_threshold]

            train_users += [train_user]
            val_users += [val_user]
            test_users += [test_user]

        self.train_loader = DataLoader(
            TouchDataset(train_users),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_data,
            num_workers=WORKER_NUM
        )
        self.val_loader = DataLoader(
            TouchDataset(val_users),
            batch_size=BATCH_SIZE,
            collate_fn=collate_data,
            num_workers=WORKER_NUM
        )
        self.test_loader = DataLoader(
            TouchDataset(test_users),
            batch_size=BATCH_SIZE,
            collate_fn=collate_data,
            num_workers=WORKER_NUM
        )
