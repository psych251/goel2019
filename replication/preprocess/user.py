import copy
import os
from typing import overload
import numpy as np

from replication.preprocess.condition import Condition

STRESSED_PREFIX = "stressed"
UNSTRESSED_PREFIX = "unstressed"


class User:
    stressed_condition: Condition
    unstressed_condition: Condition
    name: str

    def clean_data(self):
        for condition in [self.stressed_condition, self.unstressed_condition]:
            condition.clean_tasks()

    def __copy__(self):
        user = User()
        user.stressed_condition = copy.copy(self.stressed_condition)
        user.unstressed_condition = copy.copy(self.unstressed_condition)
        return user

    @overload
    def __init__(self, file_prefix: str, name: str):
        pass

    @overload
    def __init__(self):
        pass

    def normalize_data(self):
        moves_values = ["x", "y", "x_speed", "y_speed", "major_axis", "minor_axis", "contact_area"]
        conditions = [self.stressed_condition, self.unstressed_condition]
        for moves_value in moves_values:
            values = [
                getattr(entry, moves_value)
                for condition in conditions
                for task in condition.tasks
                for entry in task.track_pad_entries
            ]
            mean = np.average(values)
            var = np.std(values)
            if var != 0:
                for condition in conditions:
                    for task in condition.tasks:
                        for entry in task.track_pad_entries:
                            setattr(entry, moves_value, (getattr(entry, moves_value) - mean) / var)
                        task.populate_separated_track_pad_entries()

    def normalize_separated_track_pad_entries(self):
        moves_values = ["x", "y", "x_speed", "y_speed", "major_axis", "minor_axis", "contact_area"]
        conditions = [self.stressed_condition, self.unstressed_condition]
        for moves_value in moves_values:
            values = [
                getattr(entry, moves_value)
                for condition in conditions
                for task in condition.tasks
                for trace in task.separated_track_pad_entries
                for entry in trace
            ]
            mean = np.average(values)
            var = np.std(values)
            if var != 0:
                for condition in conditions:
                    for task in condition.tasks:
                        for trace in task.separated_track_pad_entries:
                            for entry in trace:
                                setattr(entry, moves_value, (getattr(entry, moves_value) - mean) / var)

    def __init__(self, *args):
        if len(args) == 0:
            self.stressed_condition = Condition()
            self.unstressed_condition = Condition()
            self.name = ""
            pass
        elif len(args) == 2 and isinstance(args[0], str):
            file_prefix = args[0]
            name = args[1]
            self.stressed_condition = Condition(os.path.join(file_prefix, STRESSED_PREFIX))
            self.unstressed_condition = Condition(os.path.join(file_prefix, UNSTRESSED_PREFIX))
            self.name = name
            self.normalize_data()
        else:
            raise ValueError

    def __str__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    example_user = User("../../original_data/data/raw_data/A1/", "A1")
    print(example_user)
