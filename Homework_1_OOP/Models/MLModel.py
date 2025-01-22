class MLModel:
    def fit(self, X: list[int | float] | list[list[int | float]], y: list) -> None:
        raise NotImplementedError

    def predict(self, X: list[int | float] | list[list[int | float]]):
        raise NotImplementedError

    def inherited_method(self):
        print("INHERITED METHOD EXAMPLE!")