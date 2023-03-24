import os

from scipy.stats import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .config import *


class DataProcessor:
    def __init__(self):
        self._xTrain = None
        self._yTrain = None
        self._xTest = None
        self._yTest = None

    def process(self):
        #   Read in the dataset into pandas dataframe
        try:
            polar_df = pl.read_csv(source=os.getcwd() + "\src\data\For_modeling.csv", dtypes=DTYPES).drop('')
        except FileNotFoundError:
            polar_df = pl.read_csv(source=os.getcwd() + "\data\For_modeling.csv", dtypes=DTYPES).drop('')

        polar_df = polar_df.sample(SAMPLE_SIZE)
        pandas_df = polar_df.to_pandas()

        #   Remove outliers
        df = pandas_df.drop('Duration', axis=1)
        z_scores = stats.zscore(df)
        threshold = 3
        outliers = (abs(z_scores) > threshold).any(axis=1)
        pandas_df = pandas_df[~outliers]

        #   Split the data into target and independent variables pandas data frames
        x = pandas_df.drop(TARGET_VARIABLE, axis=1)
        y = pandas_df[TARGET_VARIABLE]

        #  Split the data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        #   Normalize the data
        scaler = MinMaxScaler()
        x_train_norm = scaler.fit_transform(x_train)

        #   Get selected features from dataset
        lr = LinearRegression()

        pipe = Pipeline([
            ('rfe', RFE(lr)),
            ('lr', LinearRegression())
        ])

        parameters = {
            'rfe__n_features_to_select': range(1, len(x_train.columns) + 1)
        }

        grid = GridSearchCV(pipe, param_grid=parameters, cv=10, n_jobs=-1)
        grid.fit(x_train_norm, y_train)

        print(grid.best_params_)
        print(grid.best_score_)
        print(grid.best_estimator_.named_steps['rfe'].support_)

        #   Get feature names from the best model
        selected_features = x_train.columns[grid.best_estimator_.named_steps['rfe'].support_]

        #   Remove unneeded features from dataset
        x_train = x_train[selected_features]
        x_test = x_test[selected_features]

        #   Return split dataset
        self._xTrain, self._xTest, self._yTrain, self._yTest = x_train, x_test, y_train, y_test

    def get_split_dataset(self):
        return self._xTrain, self._xTest, self._yTrain, self._yTest

    def get_features(self) -> [str]:
        features: [str] = list(self._xTrain.columns)
        return features
