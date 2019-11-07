import bisect
from typing import List, Optional, TypeVar, overload

import pandas
import seaborn as sns

from replication.preprocess.console import ConsoleFile, Task
from replication.preprocess.cursor import CursorEntry, CursorFile
from replication.preprocess.data_entry import DataEntry
from replication.preprocess.moves import MovesFile, TrackPadEntry

MOVES_POSTFIX = "_moves.txt"
CONSOLE_POSTFIX = "_console.txt"
CURSOR_POSTFIX = "_cursor.txt"

VALID_CURSOR_COUNT = 50


class TaskMoves(Task):
    track_pad_entries: Optional[List[TrackPadEntry]]
    cursor_entries: Optional[List[CursorEntry]]

    # noinspection PyMissingConstructor
    def __init__(self, task: Task):
        self.__dict__.update(vars(task))
        self.track_pad_entries = None
        self.cursor_entries = None

    E = TypeVar('E', bound='DataEntry')

    @staticmethod
    def filter_data_entries(entries: List[E], start_time, end_time) -> List[E]:
        start_idx = bisect.bisect_left(entries, DataEntry(start_time))
        end_idx = bisect.bisect_right(entries, DataEntry(end_time))
        return entries[start_idx: end_idx]

    def populate_track_pad_entries(self, move_file: MovesFile):
        self.track_pad_entries = self.filter_data_entries(move_file.entries, self.start, self.finish)

    def populate_cursor_entries(self, move_file: CursorFile):
        self.cursor_entries = self.filter_data_entries(move_file.entries, self.start, self.finish)

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

    @overload
    def __init__(self, file_prefix):
        pass

    @overload
    def __init__(self):
        pass

    def __init__(self, *args):
        if len(args) == 0:
            pass
        elif len(args) == 1 and isinstance(args[0], str):
            file_prefix = args[0]
            console_file = ConsoleFile(file_prefix + CONSOLE_POSTFIX)
            moves_file = MovesFile(file_prefix + MOVES_POSTFIX)
            cursor_file = CursorFile(file_prefix + CURSOR_POSTFIX)
            self.tasks = [TaskMoves(entry) for entry in console_file.entries.values()]
            for task in self.tasks:
                task.populate_track_pad_entries(moves_file)
                task.populate_cursor_entries(cursor_file)
        else:
            raise ValueError

    def clean_tasks(self):
        self.tasks = list(filter(lambda task: len(task.cursor_entries) > VALID_CURSOR_COUNT, self.tasks))

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    example_condition = Condition("../../original_data/data/raw_data/A1/stressed")
    print(example_condition)
