import os

from replication.preprocess.condition import Condition

STRESSED_PREFIX = "stressed"
UNSTRESSED_PREFIX = "unstressed"


class User:
    stressed_condition: Condition
    unstressed_condition: Condition

    def __init__(self, file_prefix):
        self.stressed_condition = Condition(os.path.join(file_prefix, STRESSED_PREFIX))
        self.unstressed_condition = Condition(os.path.join(file_prefix, UNSTRESSED_PREFIX))

    def __str__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    example_user = User("../../original_data/data/raw_data/A1/")
    print(example_user)
