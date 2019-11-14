import bisect
import math
from typing import List, Optional, TypeVar, overload, Tuple

import pandas
import seaborn as sns

from replication.common import SAMPLE_INTERVAL, SAMPLE_TOLERANCE
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
    separated_track_pad_entries: Optional[List[List[TrackPadEntry]]]
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

    def populate_separated_track_pad_entries(self):
        separated_track_pad_entries: List[List[TrackPadEntry]] = []
        for entry in self.track_pad_entries:
            entered = False
            for history_entries in separated_track_pad_entries:
                if history_entries[-1].is_close(entry, SAMPLE_TOLERANCE):
                    history_entries += [entry]
                    entered = True
                    break
            if entered:
                continue
            for history_entries in separated_track_pad_entries:
                if not DataEntry.is_close(history_entries[-1], entry, SAMPLE_INTERVAL):
                    history_entries += [entry]
                    entered = True
                    break
            if entered:
                continue
            separated_track_pad_entries += [[entry]]
        self.separated_track_pad_entries = \
            self.fill_separated_data_entries(separated_track_pad_entries, SAMPLE_INTERVAL)

    @staticmethod
    def fill_separated_data_entries(data_entries: List[List[E]], interval: float) -> List[List[E]]:
        new_entries = [[] for _ in range(len(data_entries))]
        cur_indices: List[int] = [-1] * len(data_entries)

        def get_next_indices() -> Tuple[Optional[List[int]], float]:
            min_next_time = math.inf
            result_next_indices: Optional[List[int]] = None
            for entries_i, cur_index in enumerate(cur_indices):
                next_index = cur_index + 1
                if next_index < len(data_entries[entries_i]):
                    next_time = data_entries[entries_i][next_index].time
                    if next_time < min_next_time:
                        min_next_time = next_time
                        result_next_indices = cur_indices.copy()
                    if next_time == min_next_time:
                        result_next_indices[entries_i] = next_index
            return result_next_indices, min_next_time

        cur_indices, cur_time = get_next_indices()

        while cur_indices is not None:
            next_indices, next_time = get_next_indices()
            gap = (next_time - cur_time) if next_indices is not None else interval
            count = max(int(gap // interval), 1)
            interval_gap = gap / count
            for i in range(count):
                new_entry_time = cur_time + interval_gap * i
                for entries_i, cur_index in enumerate(cur_indices):
                    next_index = cur_index + 1
                    cur_entries = data_entries[entries_i]
                    cur_entry = cur_entries[cur_index] if cur_index != -1 else cur_entries[next_index]
                    next_entry = cur_entries[next_index] if next_index < len(cur_entries) else cur_entries[cur_index]
                    new_entries[entries_i] += [DataEntry.interpolate(cur_entry, next_entry, new_entry_time)]
            cur_indices, cur_time = next_indices, next_time

        return new_entries

    def populate_cursor_entries(self, move_file: CursorFile):
        self.cursor_entries = self.filter_data_entries(move_file.entries, self.start, self.finish)
        self.cursor_entries = self.fill_data_entries(self.cursor_entries, SAMPLE_INTERVAL)

    @staticmethod
    def fill_data_entries(data_entries: List[E], interval: float, threshold: float = 0.05) -> List[E]:
        new_entries = []
        for prev_entry, next_entry in zip(data_entries, data_entries[1:] + [None]):
            new_entries += [prev_entry]
            if next_entry is not None and next_entry.time - prev_entry.time > interval + threshold:
                gap = next_entry.time - prev_entry.time
                count = int(gap // interval)
                interval_gap = gap / count
                for i in range(count - 1):
                    new_entry_time = prev_entry.time + interval_gap * (i + 1)
                    new_entries += [DataEntry.interpolate(prev_entry, next_entry, new_entry_time)]
        return new_entries

    @staticmethod
    def track_pad_entries_to_df(entries: List[TrackPadEntry]) -> pandas.DataFrame:
        return pandas.DataFrame([vars(entry) for entry in entries],
                                columns=["time", "press", "pos", "x", "y", "x_speed", "y_speed", "major_axis",
                                         "minor_axis", "contact_area"])

    @staticmethod
    def cursor_entries_to_df(entries: List[CursorEntry]) -> pandas.DataFrame:
        return pandas.DataFrame([vars(entry) for entry in entries],
                                columns=["time", "x", "y"])

    @property
    def track_pad_df(self):
        return self.track_pad_entries_to_df(self.track_pad_entries)

    @property
    def separated_track_pad_df(self):
        data_frames = [self.track_pad_entries_to_df(entries)
                       for entries in self.separated_track_pad_entries]
        return pandas.concat(data_frames, keys=range(len(data_frames)))

    @property
    def cursor_df(self):
        return self.cursor_entries_to_df(self.cursor_entries)

    def draw_moves(self, axis):
        track_pad_df = self.track_pad_df
        sns.scatterplot(track_pad_df['time'], track_pad_df['x'], ax=axis)

    def draw_cursors(self, axis):
        cursor_df = self.cursor_df
        sns.scatterplot(cursor_df['time'], cursor_df['x'], ax=axis)
        sns.scatterplot(cursor_df['time'], cursor_df['y'], ax=axis)

    def draw_separated_moves(self, axis):
        cursor_df = self.cursor_df
        sns.scatterplot(cursor_df['time'], cursor_df['x'], ax=axis)
        sns.scatterplot(cursor_df['time'], cursor_df['y'], ax=axis)

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
                task.populate_separated_track_pad_entries()
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
