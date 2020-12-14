from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """
    guid: str
    words: List[str]
    metainfo: Optional[str] = None
    labels: Optional[List[str]] = None


def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return None


