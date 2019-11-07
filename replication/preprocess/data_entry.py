import math
from typing import overload, TypeVar, Tuple


class DataEntry:
    time: float
    valid: bool

    @overload
    def __init__(self, time: float):
        pass

    @overload
    def __init__(self):
        pass

    def __init__(self, *args):
        self.valid = True
        if len(args) == 0:
            return
        elif len(args) == 1:
            if isinstance(args[0], float):
                self.time = args[0]
                return
        raise TypeError("Cannot initialize DataEntry")

    E = TypeVar('E', bound='DataEntry')

    @staticmethod
    def interpolate_float(value_1: Tuple[float, float], value_2: Tuple[float, float], time: float):
        return (value_1[1] * (value_2[0] - time) + value_2[1] * (time - value_1[0])) / (value_2[0] - value_1[0])

    @staticmethod
    def interpolate(value_1: E, value_2: E, time: float) -> E:
        new_value = DataEntry(time)
        new_value.__class__ = type(value_1)
        vars_1 = vars(value_1)
        vars_2 = vars(value_2)
        for name, name_value_1 in vars_1.items():
            name_value_2 = vars_2[name]
            new_name_value = \
                DataEntry.interpolate_float((value_1.time, name_value_1), (value_2.time, name_value_2), time)
            setattr(new_value, name, new_name_value)
        assert math.fabs(time - new_value.time) < 1e-5
        new_value.valid = False
        return new_value

    def __lt__(self, other):
        if isinstance(other, DataEntry):
            return self.time < other.time
        elif isinstance(other, float):
            return self.time < other
        else:
            raise TypeError("Unsupported type!")

    def __gt__(self, other):
        if isinstance(other, DataEntry):
            return self.time > other.time
        elif isinstance(other, float):
            return self.time > other
        else:
            raise TypeError("Unsupported type!")
