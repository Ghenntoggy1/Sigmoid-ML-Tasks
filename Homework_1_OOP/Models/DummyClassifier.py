from overrides import override
from collections import Counter

from Homework_1_OOP.Models.MLModel import MLModel


class DummyClassifier(MLModel):
    def __init__(self):
        self.__mode = False

    @override
    def fit(self, X: list | list[list], y: list) -> None:
        # self.__mode = max(set(y), key=y.count)
        if set(y) != {0, 1}:
            raise ValueError("y must contain only 0 and 1 values.")
        if len(y) == 0:
            raise ValueError("y contains no values inside.")

        occurrence_count = Counter(y)
        self.__mode = occurrence_count.most_common(1)[0][0] # most_common(1) => only one most common value (still list of tuples),
                                                            # most_common(1)[0] => return the tuple (value, occurrences),
                                                            # most_common(1)[0][0] => return the most common value.

    @override
    def predict(self, X: list | list[list]) -> list[bool]:
        return [bool(self.__mode) for _ in range(len(X))]