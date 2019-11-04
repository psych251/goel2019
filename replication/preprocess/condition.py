from typing import List

import pandas
import seaborn as sns

from replication.preprocess.console import ConsoleFile, Task
from replication.preprocess.cursor import CursorEntry, CursorFile
from replication.preprocess.moves import MovesFile, TrackPadEntry

MOVES_POSTFIX = "_moves.txt"
CONSOLE_POSTFIX = "_console.txt"
CURSOR_POSTFIX = "_cursor.txt"

class TaskMoves(Task):
    track_pad_entries: List[TrackPadEntry]
    cursor_entries: List[CursorEntry]

    # noinspection PyMissingConstructor
    def __init__(self, task: Task):
        self.__dict__.update(vars(task))

    def populate_track_pad_entries(self, move_file: MovesFile):
        self.track_pad_entries = list(filter(lambda entry: self.start <= entry.time <= self.finish, move_file.entries))

    def populate_cursor_entries(self, move_file: CursorFile):
        self.cursor_entries = list(filter(lambda entry: self.start <= entry.time <= self.finish, move_file.entries))

    @staticmethod
    def track_pad_entries_to_df(entries: List[TrackPadEntry]) -> pandas.DataFrame:
        return pandas.DataFrame([vars(entry) for entry in entries],
                                columns=["time", "press", "pos", "x", "y", "x_speed", "y_speed", "major_axis",
                                         "minor_axis", "contact_area"])

    @staticmethod
    def cursor_entries_to_df(entries: List[CursorEntry]) -> pandas.DataFrame:
        return pandas.DataFrame([vars(entry) for entry in entries],
                                columns=["x", "y"])

    @property
    def track_pad_df(self):
        return self.track_pad_entries_to_df(self.track_pad_entries)

    @property
    def cursor_df(self):
        return self.cursor_entries_to_df(self.cursor_entries)

    def draw_moves(self, axis):
        track_pad_df = self.track_pad_df
        sns.scatterplot(track_pad_df['time'], track_pad_df['x'], ax=axis)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class Condition:
    tasks = List[TaskMoves]

    def __init__(self, file_prefix):
        console_file = ConsoleFile(file_prefix + CONSOLE_POSTFIX)
        moves_file = MovesFile(file_prefix + MOVES_POSTFIX)
        cursor_file = CursorFile(file_prefix + CURSOR_POSTFIX)
        self.tasks = [TaskMoves(entry) for entry in console_file.entries.values()]
        for task in self.tasks:
            task.populate_track_pad_entries(moves_file)
            task.populate_cursor_entries(cursor_file)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    example_condition = Condition("../../original_data/data/raw_data/A1/stressed")
    print(example_condition)
