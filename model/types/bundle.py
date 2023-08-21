from collections import namedtuple


Bundle = namedtuple("Bundle", ["content", "style"])
TrainBundle = namedtuple("TrainBundle", ["content", "style", "target"])
