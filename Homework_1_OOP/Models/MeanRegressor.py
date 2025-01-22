from overrides import override

from Homework_1_OOP.Models.MLModel import MLModel


class MeanRegressor(MLModel):
    def __init__(self):
        self.__mean: float = 0.0

    @override
    def fit(self, X: list[int | float] | list[list[int | float]], y: list) -> None:
        self.__mean = sum(X) / len(y)

    @override
    def predict(self, X: list[int | float] | list[list[int | float]]) -> list:
        return [self.__mean for _ in range(len(X))]