# Import Data Structures
import numpy as np
import pandas as pd

# Import Base Classes for Type Annotation
from sklearn.base import TransformerMixin, BaseEstimator
from typing import List, Tuple

# Import Structure Manipulation Methods
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import Visualization Libs
import matplotlib.pyplot as plt
import seaborn as sns
from dtreeviz import model

# Import ML Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Import Interpretation Metrics
from sklearn.metrics import (mean_squared_error, mean_absolute_error, root_mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from lime.lime_tabular import LimeTabularExplainer
import shap

# Import Time
import time
from tqdm import tqdm


electric_motor_temps_df = pd.read_csv(filepath_or_buffer="dataset/measures_v2.csv")


electric_motor_temps_df = electric_motor_temps_df.sample(n=10000, random_state=42)
electric_motor_temps_df.info()
print(electric_motor_temps_df.isnull().sum())
electric_motor_temps_df.describe()

electric_motor_temps_df.head(n=10)



X_motor_features: pd.DataFrame = electric_motor_temps_df.drop(columns=['pm'])
y_motor_target: pd.Series = electric_motor_temps_df['pm']
X_motor_train, X_motor_test, y_motor_train, y_motor_test = train_test_split(X_motor_features,
                                                                            y_motor_target,
                                                                            test_size=.2,
                                                                            random_state=42)
print(X_motor_train)
print(X_motor_test)
def create_pipeline(transformers: List[Tuple[str, TransformerMixin]] = None,
                    scaler: TransformerMixin = None,
                    model: BaseEstimator = None) -> Pipeline:
    pipeline = Pipeline(steps=[])

    if transformers is not None and len(transformers) > 0:
        pipeline.steps.extend(transformers)

    if scaler is not None:
        pipeline.steps.append(
            ('scale', scaler)
        )

    if model is not None:
        pipeline.steps.append(
            ('model', model)
        )

    return pipeline
features_to_drop = ['profile_id']
drop_transformer = ColumnTransformer(
    transformers=[
        ("drop", "drop", features_to_drop)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False,
    force_int_remainder_cols=False
).set_output(transform='pandas')
standard_scaler = StandardScaler()
standard_scaler.set_output(transform='pandas')



regression_results_table: pd.DataFrame = pd.DataFrame(columns=['Estimator', 'MSE', 'MAE', 'RMSE'])
estimators: dict[str, BaseEstimator] = {
    'Linear Regressor': LinearRegression(),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
    'Random Forest Regressor': RandomForestRegressor(verbose=1),
    'Decision Tree Regressor': DecisionTreeRegressor()
}
best_regressor_estimator = None
for estimator_name, estimator_object in tqdm(estimators.items(), desc="Fitting regression models"):
    pipeline_motor: Pipeline = create_pipeline(transformers=[('drop_transform', drop_transformer)], scaler=None, model=estimator_object)

    start_time = time.time()
    pipeline_motor.fit(X=X_motor_train, y=y_motor_train)
    end_time = time.time()
    total_time = (end_time - start_time) * 1000

    y_motor_hat = pipeline_motor.predict(X=X_motor_test)
    mse = mean_squared_error(y_true=y_motor_test, y_pred=y_motor_hat)
    mae = mean_absolute_error(y_true=y_motor_test, y_pred=y_motor_hat)
    rmse = root_mean_squared_error(y_true=y_motor_test, y_pred=y_motor_hat)
    tqdm.write(f"Estimator: {estimator_name}"
               f"\nTime Taken: {total_time:.4f} ms"
               f"\nMSE: {mse}"
               f"\nMAE: {mae}"
               f"\nRMSE: {rmse}"
               f"\n")
    regression_results_table = pd.concat([regression_results_table,
                                          pd.DataFrame(columns=['Estimator', 'MSE', 'MAE', 'RMSE'],
                                                       data=[[estimator_name, mse, mae, rmse]])],
                                         axis=0,
                                         ignore_index=True
    )
    if best_regressor_estimator is None or mse < best_regressor_estimator['MSE']:
        best_regressor_estimator = {
            'Estimator': estimator_name,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Pipeline': pipeline_motor
        }

print(regression_results_table)
print(best_regressor_estimator)


random_forest_regressor = RandomForestRegressor()
# pipeline_motor = create_pipeline(transformers=[('drop', drop_transformer)], model=random_forest_regressor)
pipeline_motor = best_regressor_estimator['Pipeline']
X_motor_train_transformed = pipeline_motor[:-1].transform(X=X_motor_train)
X_motor_test_transformed = pipeline_motor[:-1].transform(X=X_motor_test)
explainer = LimeTabularExplainer(
    training_data=X_motor_train_transformed.values,
    feature_names=X_motor_train_transformed.columns.tolist(),
    class_names=[y_motor_target.name],
    verbose=True,
    mode='regression'
)
explanation = explainer.explain_instance(X_motor_test_transformed.iloc[2], pipeline_motor[-1].predict)

import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Load your model, explainer, and one sample

st.set_page_config(layout="wide")
st.title("LIME Explanation")

components = explanation.as_html()
st.components.v1.html(components, height=800, scrolling=True)
