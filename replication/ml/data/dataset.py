import _bisect

import torch
import torch.utils.data
from typing import List, Tuple

from replication.preprocess.cursor import CursorEntry
from replication.preprocess.moves import TrackPadEntry
from replication.preprocess.user import User


class TouchDataset(torch.utils.data.Dataset):
    valid_user_tasks: List[Tuple[int]]
    valid_user_combinations: List[int]
    total_combinations: int
    users: List[User]

    def __init__(self, users: List[User]):
        self.valid_user_tasks = []
        self.valid_user_combinations = []
        self.total_combinations = 0
        for user in users:
            stressed_count = len(user.stressed_condition.tasks)
            unstressed_count = len(user.unstressed_condition.tasks)
            self.valid_user_tasks += [(stressed_count, unstressed_count)]
            new_combinations = stressed_count * unstressed_count
            self.valid_user_combinations += [self.total_combinations]
            self.total_combinations += new_combinations
        self.users = users

    @staticmethod
    def cursor_to_list(entries: List[CursorEntry]):
        x = []
        y = []
        t = []
        for entry in entries:
            x += [entry.x]
            y += [entry.y]
            t += [1.0 if entry.valid else 0.0]
        return x, y, t

    @staticmethod
    def track_pad_to_list(entries: List[TrackPadEntry]):
        x = []
        y = []
        major = []
        minor = []
        area = []
        valid = []
        press = []
        for entry in entries:
            x += [entry.x]
            y += [entry.y]
            valid += [1.0 if entry.valid else 0.0]
            major += [entry.major_axis]
            minor += [entry.minor_axis]
            area += [entry.contact_area]
            press += [entry.press]
        return x, y, major, minor, valid, press, area

    @staticmethod
    def separated_track_pad_to_list(entries_list: List[List[TrackPadEntry]]):
        return [TouchDataset.track_pad_to_list(entries) for entries in entries_list]

    def __getitem__(self, index):
        user_id = _bisect.bisect_right(self.valid_user_combinations, index)
        user_id -= 1
        user = self.users[user_id]

        task_index = index - self.valid_user_combinations[user_id]
        stressed_task_count = len(user.stressed_condition.tasks)
        stressed_task_id = task_index % stressed_task_count
        unstressed_task_count = len(user.unstressed_condition.tasks)
        unstressed_task_id = task_index // stressed_task_count
        stressed_task = user.stressed_condition.tasks[stressed_task_id]
        unstressed_task = user.unstressed_condition.tasks[unstressed_task_id]

        # noinspection PyArgumentList
        stressed_tensor = torch.Tensor(self.separated_track_pad_to_list(stressed_task.separated_track_pad_entries))
        # noinspection PyArgumentList
        unstressed_tensor = torch.Tensor(self.separated_track_pad_to_list(unstressed_task.separated_track_pad_entries))
        return (
                   self.users[user_id].name,
                   stressed_task_id / stressed_task_count,
                   unstressed_task_id / unstressed_task_count
               ), (stressed_tensor, unstressed_tensor)

    def __len__(self):
        return self.total_combinations
