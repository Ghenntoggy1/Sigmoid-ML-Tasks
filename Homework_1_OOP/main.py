from Homework_1_OOP.Models.Dataset import Dataset
from Homework_1_OOP.Models.DummyClassifier import DummyClassifier
from Homework_1_OOP.Models.MeanRegressor import MeanRegressor

if __name__ == '__main__':
    X: list = [5, 2, 1, 4, 6, 7, 4, 1, 2, 3] #  split training will have 6
    y: list = [0, 1, 0, 0, 1, 0, 1, 1, 1, 1]

    dataset: Dataset = Dataset(features=X, targets=y)
    (X_train, y_train), (X_test, y_test) = dataset.split() # default value - 0.8

    print(f"X_train = {X_train}")
    print(f"y_train = {y_train}")
    print(f"X_test = {X_test}")
    print(f"y_test = {y_test}")

    classifier_model: DummyClassifier = DummyClassifier()
    classifier_model.fit(X=X_train, y=y_train)
    print(f"Predicted classes: {classifier_model.predict(X=X_test)}")

    regression_model: MeanRegressor = MeanRegressor()
    regression_model.fit(X=X_train, y=y_train)
    print(f"Predicted values: {regression_model.predict(X=X_test)}")


