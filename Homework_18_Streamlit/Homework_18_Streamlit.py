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

# Import Low-Level operations
import os
import io

# Import Visualization Libs
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Import ML Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Import Interpretation Metrics
from sklearn.metrics import (mean_squared_error, mean_absolute_error, root_mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def load_image() -> str:
    image_file_path = os.path.join(current_directory, 'streamlit_files', 'heart.png')
    return image_file_path

def load_dataset() -> pd.DataFrame:
    if 'heart_disease_dataset' not in st.session_state:
        dataset_path = os.path.join(current_directory, 'dataset', 'heart.csv')
        heart_disease_dataset: pd.DataFrame = pd.read_csv(dataset_path)
        st.session_state.heart_disease_dataset = heart_disease_dataset

    return st.session_state.heart_disease_dataset

def home_page() -> None:
    st.title("Heart Disease Dataset")

    image = load_image()
    st.image(image=image, use_container_width=True)

    st.header("Dataset Description")
    st.write("Heart Disease Dataset is a dataset that contains information about patients with Coronary "
             "Artery Disease (CAD), with detailed information about their biometric information, as well "
             "as regular people without the disease.")
    st.write("This set is primarily used for Classification - predicting the heart failure of a patient"
             "based on their physiological data.")

    st.header("Features Description")
    st.write("1. Age: age of the patient (years)")
    st.write("2. Sex: sex of the patient (M: Male, F: Female)")
    st.write("3. ChestPainType: chest pain type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, "
             "ASY: Asymptomatic)")
    st.write("4. RestingBP: resting blood pressure (mm Hg)")
    st.write("5. Cholesterol: serum cholesterol (mm/dl)")
    st.write("6. FastingBS: fasting blood sugar (1: if FastingBS > 120 mg/dl, 0: otherwise)")
    st.write("7. RestingECG: resting electrocardiogram results (Normal: Normal, ST: having ST-T wave abnormality "
             "(T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite "
             "left ventricular hypertrophy by Estes' criteria)")
    st.write("8. MaxHR: maximum heart rate achieved (Numeric value between 60 and 202)")
    st.write("9. ExerciseAngina: exercise-induced angina (Y: Yes, N: No)")
    st.write("10. Oldpeak: oldpeak = ST (Numeric value measured in depression)")
    st.write("11. ST_Slope: the slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping)")
    st.write("12. HeartDisease: output class (1: heart disease, 0: Normal)")

    st.header("Dataset Link")
    st.write("Dataset is available here: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/"
             "fedesoriano/heart-failure-prediction)")

def get_df_info(df: pd.DataFrame) -> pd.DataFrame:
    buffer = io.StringIO()
    df.info(buf=buffer)
    lines = buffer.getvalue().split('\n')
    lines_to_print = [1, 2]
    for i in lines_to_print:
        st.write(lines[i])
    list_of_list = []
    for x in lines[5:-3]:
        list = x.split()
        list_of_list.append(list)
    info_df = pd.DataFrame(list_of_list, columns=['index', 'Column', 'Non-null-Count', 'null', 'Dtype'])
    info_df.drop(columns=['index', 'null'], axis=1, inplace=True)
    return info_df

def dataset_exploration() -> None:
    st.markdown("# Dataset Exploration")
    heart_disease_dataset: pd.DataFrame = load_dataset()

    st.markdown('### Data Examples')
    st.dataframe(heart_disease_dataset.head(n=10))

    st.markdown('### Dataset Information')
    st.dataframe(get_df_info(df=heart_disease_dataset))

    target_column: str = "HeartDisease"

    st.markdown(
        f"As it may be noticed, there are **{heart_disease_dataset.shape[0]}** data samples, as well as "
        f"**{heart_disease_dataset.columns.shape[0] - 1}** features, and **1** target column - `{target_column}`."
    )

    st.markdown("### Dataset Basic Null Values Check")
    st.write(heart_disease_dataset.isnull().sum())

    st.markdown("Also, no null data is visible in the columns. However, there are multiple `object` data type columns,"
                "which can include Null values in format of empty strings or any other way different from"
                "`numpy` or `pandas` null values.")

    st.markdown("### Dataset Object Column Null Values Check")
    st.markdown("Object Columns - Unique Values:")

    categorical_columns = []
    continuous_columns = [col for col in heart_disease_dataset.columns if col != 'HeartDisease']
    for column in heart_disease_dataset.columns:
        if heart_disease_dataset[column].dtype == 'object':
            categorical_columns.append(column)
            column_1, column_2 = st.columns(2)
            column_1.markdown(f'{column}')
            column_2.markdown(f'{heart_disease_dataset[column].unique()}')
    continuous_columns = [col for col in continuous_columns if col not in categorical_columns]

    st.markdown("Now, there is certainty that there are no Null values in object columns, as well as numerical "
                "columns, therefore, no missing values imputation should be done.")

    st.markdown("### Basic Dataset Visualization")
    st.markdown("#### Count plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x=heart_disease_dataset['HeartDisease'])

    ax.bar_label(ax.containers[0])
    ax.set_title('Count of Patients with Heart Disease vs without')
    ax.set_xlabel('Heart Disease')
    ax.set_ylabel('Count')

    st.pyplot(fig)
    st.markdown("As it may be noticed, the distribution of positive and negative target classes is close to each"
                "other, which means that there is no case of imbalanced dataset. Slight prevalence of positive cases"
                "is still present, but it is not very significant.")

    st.markdown("#### Distribution Plots")
    features_list = heart_disease_dataset.drop(columns=['HeartDisease']).columns
    n_cols = 3
    n_rows = (features_list.shape[0] + n_cols - 1) // n_cols
    fig, axes = plt.subplots(figsize=(10, 10), nrows=n_rows, ncols=n_cols)
    axes = axes.flatten()
    for idx, column in enumerate(heart_disease_dataset.drop(columns=['HeartDisease']).columns):
        sns.histplot(data=heart_disease_dataset, x=column, kde=True, ax=axes[idx])
        axes[idx].set_title(f"Distribution of {column}")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("As a result, continuous-valued columns are very close to the Gaussian Normal Distribution, which"
                "is useful for model training, specifically for scaling of the features. At the same time, there are"
                "several discrete-valued columns, such as: `ChestPainType` or `FastingBS`, that will be encoded using "
                "One Hot Encoder at the step of model training")

    st.markdown("#### Box Plots")
    n_cols = 3
    n_rows = (len(continuous_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(figsize=(10, 5), nrows=n_rows, ncols=n_cols)
    axes = axes.flatten()
    for idx, column in enumerate(continuous_columns):
        sns.boxplot(data=heart_disease_dataset, x=column, ax=axes[idx])
        axes[idx].set_title(f"Distribution of {column}")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("Using box plots, it is possible to identify outliers in the features. In this case, there are"
                "significant number of outliers in the columns of `RestingBP`, `Cholesterol` and `OldPeak`."
                "However, in the columns of `Age` and `MaxHR` are almost no outlier at all. There are models,"
                "that are robust to outliers, such as Tree-based algorithms, which will be used in the following"
                "page.")

def model_training() -> None:
    heart_disease_dataset: pd.DataFrame = load_dataset()
    st.html("<h1>Machine Learning Model Training</h1>")
    st.html("<h2>Model Selected: Random Forest</h2>")
    st.html("<p>Decision-Tree Based ML Algorithm, that uses an ensemble of decision trees to make predicting by "
            "building multiple decision trees on random subsets of features and data sample points and combining "
            "their individual predictions/decisions into a single one, depending on the problem type:</p>")
    st.latex(r"\text{Regression:} \quad \hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(x)")
    st.latex(r"\text{Classification:} \quad \hat{y} = \mathrm{mode}\left( h_1(x), h_2(x), \dots, h_T(x) \right)")

    st.markdown(r"""
    **Legend**:
    - $\hat{y}$ – the predicted continuous or class label value  
    - $T$ – the total number of trees in the forest  
    - $h_t(x)$ – the prediction of the $t^{\text{th}}$ decision tree for input $x$
    """)

    st.header("Split the Dataset")
    st.subheader("Split into Features and Target Columns")
    X_features = heart_disease_dataset.drop(columns=['HeartDisease'])
    y_target = heart_disease_dataset['HeartDisease']
    st.markdown("##### Features:")
    st.dataframe(X_features.head(n=10))
    st.markdown("##### Target:")
    st.dataframe(y_target.head(n=10), width=200)

    st.subheader("Split into Training and Test Subsets")
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=.2, random_state=42)
    st.markdown("##### Training Subset of Features:")
    st.dataframe(X_train.head(n=10))
    st.markdown("##### Test Subset of Features:")
    st.dataframe(X_test.head(n=10))
    column_1, column_2 = st.columns(2)
    column_1.markdown("##### Training Subset of Targets:")
    column_1.dataframe(y_train.head(n=10), width=200)
    column_2.markdown("##### Test Subset of Targets:")
    column_2.dataframe(y_test.head(n=10), width=200)

    st.header("Dataset Preprocessing")
    st.subheader("One-Hot Encoding")
    categorical_columns = []
    for column in heart_disease_dataset.columns:
        if heart_disease_dataset[column].dtype == 'object':
            categorical_columns.append(column)
    X_train_cat = X_train[categorical_columns]
    X_test_cat = X_test[categorical_columns]

    X_train_num = X_train.drop(columns=categorical_columns)
    X_test_num = X_test.drop(columns=categorical_columns)

    one_hot_encoder: OneHotEncoder = OneHotEncoder(drop='first', sparse_output=False, dtype=int)
    X_train_cat_encoded = one_hot_encoder.fit_transform(X=X_train_cat, y=y_train)
    X_test_cat_encoded = one_hot_encoder.transform(X=X_test_cat)

    encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_columns)

    X_train_cat_encoded_df = pd.DataFrame(X_train_cat_encoded, columns=encoded_feature_names, index=X_train.index)
    X_test_cat_encoded_df = pd.DataFrame(X_test_cat_encoded, columns=encoded_feature_names, index=X_test.index)

    X_train = pd.concat([X_train_num, X_train_cat_encoded_df], axis=1)
    X_test = pd.concat([X_test_num, X_test_cat_encoded_df], axis=1)

    st.markdown("#### Subsets after One-Hot Encoding")
    st.markdown("##### Training Subset of Features:")
    st.dataframe(X_train.head(n=10))
    st.markdown("##### Test Subset of Features:")
    st.dataframe(X_test.head(n=10))

    st.markdown("#### Correlation Matrix")
    corr_matrix =pd.concat([X_train, y_train], axis=1).corr()
    st.dataframe(corr_matrix)

    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True)
    st.pyplot(fig)

    st.subheader("Feature Scaling")
    st.write("Standardization is another scaling technique where the values are centered around the mean with a "
             "unit standard deviation. This means that the mean of the attribute becomes zero and the resultant "
             "distribution has a unit standard deviation. This feature scaling technique is used specifically"
             "when data is normally distributed and has outliers, however this is not the only case when "
             "standardization is used.")
    st.latex(r"Z = \frac{X-\mu}{\sigma}")

    standard_scaler: StandardScaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X=X_train, y=y_train)
    X_test_scaled = standard_scaler.transform(X=X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    st.markdown("#### Subsets after Feature Scaling")
    st.markdown("##### Training Subset of Features:")
    st.dataframe(X_train_scaled_df.head(n=10))
    st.dataframe(X_train_scaled_df.describe())
    st.markdown("##### Test Subset of Features:")
    st.dataframe(X_test_scaled_df.head(n=10))
    st.dataframe(X_test_scaled_df.describe())

    st.subheader("Model Parameters Selection")
    st.markdown('##### Input Parameters')
    features = X_train_scaled_df.columns.tolist()
    selected_features = st.multiselect('Select features', features, features)
    is_default_desired = st.checkbox('Select Default Parameters', value=True)
    if is_default_desired:
        n_estimators = 100
        criterion = 'gini'
        max_depth = None
        min_samples_split = 2
        min_samples_leaf = 1
        max_leaf_nodes = 50
        random_state = 42
    else:
        n_estimators = st.slider('Number of Estimators',
                                 min_value=10,
                                 max_value=1000,
                                 value=100,
                                 help='The number of trees in the forest'
                                 )
        criterion = st.selectbox('Criterion', ['gini', 'entropy', 'log_loss'],
                                 index=0,
                                 help='The function to measure the quality of a split'
                                 )
        min_samples_split = st.slider('Min Samples Split',
                                      min_value=2, max_value=30, value=2,
                                      help='The minimum number of samples required to split an internal node')
        min_samples_leaf = st.slider('Min Samples Leaf',
                                     min_value=1, max_value=20, value=1,
                                     help='The minimum number of samples required to be at a leaf node')
        max_leaf_nodes = st.slider('Max Leaf Nodes',
                                   min_value=10, max_value=100, value=50,
                                   help='Grow trees with max_leaf_nodes in best-first fashion')
        max_depth_option = st.radio('Max Depth',
                                    ['None', 'Integer'],
                                    key='max_depth_option',
                                    horizontal=True
                                    )
        if max_depth_option == 'None':
            max_depth = None
        else:
            max_depth = st.slider('Max Depth',
                                  min_value=2,
                                  max_value=100,
                                  value=10,
                                  help='The maximum depth of the tree'
                                  )

        random_state = st.number_input(
            'Random State',
            value=42,
            help='The seed used by the random number generator'
        )

    st.header('Model Training')
    if st.button('Train Random Forest Classifier'):
        X_train_fp = X_train_scaled_df[selected_features]
        X_test_fp = X_test_scaled_df[selected_features]

        random_forest_classifier: RandomForestClassifier = RandomForestClassifier(
            max_depth=max_depth,
            criterion=criterion,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf =min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            random_state =random_state,
        )

        random_forest_classifier.fit(X=X_train_fp, y=y_train)
        y_hat = random_forest_classifier.predict(X=X_test_fp)

        with st.container():
            st.header('Model Selected Parameters')
            st.write('Selected columns:', selected_features)
            st.write('Max Depth:', max_depth)
            st.write('Number of Estimators:', n_estimators)
            st.write('Criterion:', criterion)
            st.write('Min Samples Split:', min_samples_split)
            st.write('Min Samples Leaf:', min_samples_leaf)
            st.write('Max Leaf Nodes:', max_leaf_nodes)
            st.write('Random State:', random_state)

        st.subheader("Model Evaluation")
        st.write(f'Random Forest Classifier - Accuracy Score: {accuracy_score(y_true=y_test, y_pred=y_hat)}')
        st.text(classification_report(y_true=y_test, y_pred=y_hat))

        cm = confusion_matrix(y_test, y_hat)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        ax.set_title('Random Forest Classifier Confusion Matrix')
        ax.grid(False)

        st.pyplot(fig)

def navigation() -> None:
    pages = {
        "Navigation": [
            st.Page(home_page, title="Home", icon="🏠"),
            st.Page(dataset_exploration, title="Dataset Exploration", icon="📊"),
            st.Page(model_training, title="Model Training", icon="🧠"),
        ]
    }

    pg_nav = st.navigation(pages)

    pg_nav.run()

def main():
    st.set_page_config(page_title='Heart Disease Prediction', page_icon=':heart:')


    navigation()

if __name__ == '__main__':
    main()