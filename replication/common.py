from __future__ import annotations

from enum import Enum

SAMPLE_INTERVAL = 0.008
SAMPLE_TOLERANCE = 0.004

class TaskType(Enum):
    CLICK = 0
    DRAG = 1
    STEER = 2

    @staticmethod
    def from_str(str_task: str) -> TaskType:
        if str_task == 'click':
            return TaskType.CLICK
        elif str_task == 'drag':
            return TaskType.DRAG
        elif str_task == 'steer':
            return TaskType.STEER
        else:
            raise Exception(f'Cannot find task with name of {str_task}!')
