from typing import List

from replication.preprocess.data_entry import DataEntry


class CursorEntry(DataEntry):
    state: int
    x: float
    y: float

    def __init__(self, time, state, x, y):
        super().__init__(time)
        self.state = state
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class CursorFile:
    entries: List[CursorEntry]

    def parse_entry(self, entry: str):
        state_str, time_str, x_str, y_str = entry.split(',')

        time = float(time_str)
        x = float(x_str)
        y = float(y_str)
        state = int(state_str)

        self.entries += [CursorEntry(time, state, x, y)]

    def __init__(self, file_path: str):
        self.entries = []
        with open(file_path, 'r') as console_file:
            content = console_file.read()
        for line in content.splitlines():
            self.parse_entry(line)

    def __str__(self):
        return str(self.entries)


if __name__ == '__main__':
    example_moves = CursorFile("../../original_data/data/raw_data/A1/stressed_cursor.txt")
    print(example_moves)
