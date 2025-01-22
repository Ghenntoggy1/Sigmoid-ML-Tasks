from overrides import override
from collections import Counter

from Homework_1_OOP.Models.MLModel import MLModel


class DummyClassifier(MLModel):
    def __init__(self):
        self.__mode: int = 0

    @override
    def fit(self, X: list[int | float] | list[list[int | float]], y: list) -> None:
        if set(y) != {0, 1}:
            raise ValueError(f"y must contain only 0 and 1 values - set(y) = {set(y)}")
        if len(y) == 0:
            raise ValueError("y contains no values inside.")
        if len(X) != len(y):
            raise ValueError(f"X and y must be of the same length - len(X) = {len(X)} and len(y) = {len(y)}.")

        occurrence_count = Counter(y)
        self.__mode = occurrence_count.most_common(1)[0][0] # most_common(1) => only one most common value (still list of tuples),
                                                            # most_common(1)[0] => return the tuple (value, occurrences),
                                                            # most_common(1)[0][0] => return the most common value.
    @override
    def predict(self, X: list[int | float] | list[list[int | float]]) -> list:
        return [self.__mode for _ in range(len(X))]