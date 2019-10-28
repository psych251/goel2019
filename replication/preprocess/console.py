from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple, Dict

from replication.common import TaskType


class TaskEvent(Enum):
    START = 0
    FINISH = 1

    @staticmethod
    def from_str(str_event: str) -> TaskEvent:
        if str_event == 'start':
            return TaskEvent.START
        elif str_event == 'finish':
            return TaskEvent.FINISH
        else:
            raise Exception(f'Cannot find event type with name of {str_event}!')


class Task:
    page_name: str
    task_type: TaskType
    width: int
    height: int
    start: float
    finish: float

    def __init__(self, page_name: str, task_type: TaskType, width: int, height: int):
        self.page_name = page_name
        self.task_type = task_type
        self.width = width
        self.height = height

    def check(self, page_name: str, task_type: TaskType, width: int, height: int):
        assert self.page_name == page_name and \
               self.task_type == task_type and \
               self.width == width and \
               self.height == height

    def assign(self, event: TaskEvent, time: float):
        if event == TaskEvent.START:
            self.start = time
        elif event == TaskEvent.FINISH:
            self.finish = time

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class ConsoleFile:
    entries: Dict[str, Task]

    @staticmethod
    def check_px(px_str: str):
        assert px_str[-2:] == 'px'

    def parse_entry(self, entry: str):
        if "Navigated" in entry:
            return
        else:
            page_name = entry.split(':')[0]
            page_name_stripped = page_name.split('.')[0]
            if not page_name_stripped.isnumeric():  # not an actual task page
                return
            output = entry.split(' ')[1]
            time_str, task_type_str, width_str, height_str, event_str = output.split(',')
            time = float(time_str)
            event = TaskEvent.from_str(event_str)
            task_type = TaskType.from_str(task_type_str)
            self.check_px(width_str)
            self.check_px(height_str)
            width = int(width_str[:-2])
            height = int(height_str[:-2])
            if page_name in self.entries:
                self.entries[page_name].check(page_name, task_type, width, height)
                self.entries[page_name].assign(event, time)
            else:
                self.entries[page_name] = Task(page_name, task_type, width, height)
                self.entries[page_name].assign(event, time)

    def __init__(self, file_path: str):
        self.entries = dict()
        with open(file_path, 'r') as console_file:
            content = console_file.read()
        for line in content.splitlines():
            self.parse_entry(line)

    def __str__(self):
        return str(self.entries.values())


if __name__ == '__main__':
    example_console = ConsoleFile("../../original_data/data/raw_data/A1/stressed_console.txt")
    print(example_console)
