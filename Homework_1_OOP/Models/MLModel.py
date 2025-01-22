class MLModel:
    def fit(self, X: list | list[list], y: list) -> None:
        raise NotImplementedError

    def predict(self, X: list | list[list]):
        raise NotImplementedError