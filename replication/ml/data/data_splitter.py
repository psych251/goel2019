import random

import copy
from torch.utils.data import DataLoader
from typing import List, Dict

from replication.ml.data.dataset import TouchDataset
from replication.ml.params import BATCH_SIZE, WORKER_NUM
from replication.preprocess.user import User


def collate_data(array):
    return array


def match(a: int, b: int):
    return a == -1 or b == -1 or a == b


VAL_RATIO = 0.1
TRAIN_RATIO = 1 - VAL_RATIO


class DataSplitter:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Dict[str, DataLoader]
    user_names: List[str]

    def __init__(self, users: List[User]):
        train_users = []
        val_users = []
        test_users = copy.copy(users)
        self.user_names = [user.name for user in users]

        for user in users:
            for condition in [user.unstressed_condition, user.stressed_condition]:
                task_count = len(condition.tasks)
                for task_id, task in enumerate(condition.tasks):
                    task.per = task_id / task_count

        for user in users:
            train_user = User()
            val_user = User()

            train_user.name = user.name
            val_user.name = user.name

            stressed_tasks = copy.copy(user.stressed_condition.tasks)
            # random.shuffle(stressed_tasks)
            stressed_train = len(stressed_tasks) * TRAIN_RATIO
            train_threshold = int(stressed_train)
            stressed_val = len(stressed_tasks) * VAL_RATIO
            val_threshold = int(stressed_train + stressed_val)
            train_user.stressed_condition.tasks = \
                stressed_tasks[0: train_threshold]
            val_user.stressed_condition.tasks = \
                stressed_tasks[train_threshold: val_threshold]

            unstressed_tasks = copy.copy(user.unstressed_condition.tasks)
            # random.shuffle(unstressed_tasks)
            unstressed_train = len(unstressed_tasks) * TRAIN_RATIO
            train_threshold = int(unstressed_train)
            unstressed_val = len(unstressed_tasks) * VAL_RATIO
            val_threshold = int(unstressed_train + unstressed_val)
            train_user.unstressed_condition.tasks = \
                unstressed_tasks[0: train_threshold]
            val_user.unstressed_condition.tasks = \
                unstressed_tasks[train_threshold: val_threshold]

            train_users += [train_user]
            val_users += [val_user]

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
        self.test_loader = {user.name: DataLoader(
            TouchDataset([user]),
            batch_size=BATCH_SIZE,
            collate_fn=collate_data,
            num_workers=WORKER_NUM
        ) for user in test_users}
