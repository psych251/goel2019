from typing import List

from replication.preprocess.console import ConsoleFile, Task
from replication.preprocess.moves import MovesFile, TrackPadEntry

MOVES_POSTFIX = "_moves.txt"
CONSOLE_POSTFIX = "_console.txt"


class TaskMoves(Task):
    track_pad_entries: List[TrackPadEntry]

    # noinspection PyMissingConstructor
    def __init__(self, task: Task):
        self.__dict__.update(vars(task))

    def populate_entries(self, move_file: MovesFile):
        self.track_pad_entries = list(filter(lambda entry: self.start <= entry.time <= self.finish, move_file.entries))

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class Condition:
    tasks = List[TaskMoves]

    def __init__(self, file_prefix):
        console_file = ConsoleFile(file_prefix + CONSOLE_POSTFIX)
        moves_file = MovesFile(file_prefix + MOVES_POSTFIX)
        self.tasks = [TaskMoves(entry) for entry in console_file.entries.values()]
        for task in self.tasks:
            task.populate_entries(moves_file)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    example_condition = Condition("../../original_data/data/raw_data/A1/stressed")
    print(example_condition)
