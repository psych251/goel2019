import os
from typing import overload

from replication.preprocess.condition import Condition

STRESSED_PREFIX = "stressed"
UNSTRESSED_PREFIX = "unstressed"


class User:
    stressed_condition: Condition
    unstressed_condition: Condition

    def clean_data(self):
        for condition in [self.stressed_condition, self.unstressed_condition]:
            condition.clean_tasks()

    @overload
    def __init__(self, file_prefix):
        pass

    @overload
    def __init__(self):
        pass

    def __init__(self, *args):
        if len(args) == 0:
            self.stressed_condition = Condition()
            self.unstressed_condition = Condition()
            pass
        elif len(args) == 1 and isinstance(args[0], str):
            file_prefix = args[0]
            self.stressed_condition = Condition(os.path.join(file_prefix, STRESSED_PREFIX))
            self.unstressed_condition = Condition(os.path.join(file_prefix, UNSTRESSED_PREFIX))
        else:
            raise ValueError

    def __str__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    example_user = User("../../original_data/data/raw_data/A1/")
    print(example_user)
