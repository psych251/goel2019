from typing import overload


class DataEntry:
    time: float

    @overload
    def __init__(self, time: float):
        pass

    @overload
    def __init__(self):
        pass

    def __init__(self, *args):
        if len(args) == 0:
            return
        elif len(args) == 1:
            if isinstance(args[0], float):
                self.time = args[0]
                return
        raise TypeError("Cannot initialize DataEntry")

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
