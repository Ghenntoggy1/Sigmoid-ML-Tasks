import math


class Dataset:
    def __init__(self, features: list | list[list], targets: list):
        if len(features) != len(targets):
            raise ValueError(f"X and y must be of the same length - len(X) = {len(features)} and len(y) = {len(targets)}")
        self.__features = features
        self.__targets = targets

    def get_data(self) -> tuple[list, list]:
        return self.__features, self.__targets

    def split(self, train_fraction: float = 0.8) -> tuple[tuple[list, list], tuple[list, list]]:
        if not (0.0 <= train_fraction <= 1.0):
            raise ValueError(f"Train Fraction should be in range from 0.0 to 1.0 - train_fraction = {train_fraction}")

        slice_index: int = math.ceil(len(self.__features) * train_fraction)

        X_train: list = self.__features[:slice_index]
        y_train: list = self.__targets[:slice_index]
        X_test: list = self.__features[slice_index:]
        y_test: list = self.__targets[slice_index:]

        return (X_train, y_train), (X_test, y_test)