from overrides import override

from Homework_1_OOP.Models.MLModel import MLModel


class MeanRegressor(MLModel):
    def __init__(self):
        self.__mean: float = 0.0

    @override
    def fit(self, X: list[int | float] | list[list[int | float]], y: list) -> None:
        if len(X) != len(y):
            raise ValueError(f"X and y must be of the same length - len(X) = {len(X)} and len(y) = {len(y)}")

        self.__mean = sum(y) / len(y)

    @override
    def predict(self, X: list[int | float] | list[list[int | float]]) -> list:
        return [self.__mean for _ in range(len(X))]