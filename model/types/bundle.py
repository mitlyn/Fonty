from collections import namedtuple

Bundle = namedtuple("Bundle", ["content", "style", "panose"])
TrainBundle = namedtuple("TrainBundle", ["content", "style", "target", "panose"])
