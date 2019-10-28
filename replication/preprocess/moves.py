from typing import List


class TrackPadEntry:
    time: float
    press: int
    pos: int
    x: float
    y: float
    x_speed: float
    y_speed: float
    major_axis: float
    minor_axis: float
    contact_area: float

    def __init__(self, time, press, pos, x, y, x_speed, y_speed, major_axis, minor_axis, contact_area):
        self.time = time
        self.press = press
        self.pos = pos
        self.x = x
        self.y = y
        self.x_speed = x_speed
        self.y_speed = y_speed
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.contact_area = contact_area

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class MovesFile:
    entries: List[TrackPadEntry]

    def parse_entry(self, entry: str):
        time_str, press_str, pos_str, \
            x_str, y_str, \
            x_speed_str, y_speed_str, \
            major_axis_str, minor_axis_str, \
            contact_area_str = entry.split(',')

        time = float(time_str)
        press = float(press_str)
        pos = float(pos_str)
        x = float(x_str)
        y = float(y_str)
        x_speed = float(x_speed_str)
        y_speed = float(y_speed_str)
        major_axis = float(major_axis_str)
        minor_axis = float(minor_axis_str)
        contact_area = float(contact_area_str)

        self.entries += [TrackPadEntry(time, press, pos, x, y, x_speed, y_speed, major_axis, minor_axis, contact_area)]

    def __init__(self, file_path: str):
        self.entries = []
        with open(file_path, 'r') as console_file:
            content = console_file.read()
        for line in content.splitlines():
            self.parse_entry(line)

    def __str__(self):
        return str(self.entries)


if __name__ == '__main__':
    example_moves = MovesFile("../../original_data/data/raw_data/A1/stressed_moves.txt")
    print(example_moves)
