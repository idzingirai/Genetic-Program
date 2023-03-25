import os

import pandas as pd
import polars as pl

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

from config import TEST_SIZE, RANDOM_STATE, TARGET_VARIABLE, NUM_OF_FEATURES


class DataProcessor:
    def __init__(self):
        self.df = None
        self.x = None
        self.y = None

    def create_dataframe(self) -> None:
        if os.name == 'nt':
            try:
                polar_df = pl.read_csv(os.path.join(os.getcwd(), "For_modeling.csv")).drop('')
            except FileNotFoundError:
                polar_df = pl.read_csv(os.path.join(os.getcwd(), "src\\", 'For_modeling.csv')).drop('')
        else:
            try:
                polar_df = pl.read_csv(os.path.join(os.getcwd(), "For_modeling.csv")).drop('')
            except FileNotFoundError:
                polar_df = pl.read_csv(os.path.join(os.getcwd(), "src/", 'For_modeling.csv')).drop('')

        polar_df = polar_df.drop_nulls()

        sample_size = 3000
        polar_df = polar_df.sample(sample_size, with_replacement=True)

        self.df = polar_df.to_pandas()

    def remove_outliers(self) -> None:
        df = self.df.drop(TARGET_VARIABLE, axis=1)
        z_scores = stats.zscore(df)
        threshold = 2
        outliers = (abs(z_scores) > threshold).any(axis=1)
        self.df = self.df[~outliers]

    def normalize_features(self) -> None:
        scaler = MinMaxScaler()
        x_normalized = scaler.fit_transform(self.x)
        self.x = pd.DataFrame(x_normalized, columns=self.x.columns)

    def feature_selection(self) -> None:
        selector = SelectKBest(f_regression, k=NUM_OF_FEATURES)
        selector.fit_transform(self.x, self.y)
        selected_features = self.x.columns[selector.get_support()]

        self.x = self.x[selected_features]

    def process(self) -> None:
        self.create_dataframe()
        self.remove_outliers()

        self.x = self.df.drop(TARGET_VARIABLE, axis=1)
        self.y = self.df[TARGET_VARIABLE]

        self.normalize_features()
        self.feature_selection()

    def get_data(self) -> tuple:
        return train_test_split(self.x, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def get_features(self) -> list:
        return list(self.x.columns)
