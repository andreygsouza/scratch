import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ModelProcess:
    def __init__(self, n_components: int = 5, n_clusters: int = 4):
        """
        Initialize the ModelProcess class.

        Parameters
        ----------
        n_components : int
            The number of components to keep in PCA.
        n_clusters : int
            The number of clusters for KMeans.
        """
        self.outliers_columns = ["year_birth", "income"]
        self.feature_columns = [
            "education",
            "marital_status",
            "age_group",
            "preferred_channel",
            "monthly_income",
            "mnt_month_avg_wines",
            "mnt_month_avg_fruits",
            "mnt_month_avg_meat_products",
            "mnt_month_avg_fish_products",
            "mnt_month_avg_sweet_products",
            "mnt_month_avg_gold_prods",
            "mnt_month_avg_purchases",
            "num_deals_purchases",
            "num_web_visits_month",
            "num_accepted_campaigns",
            "num_children",
            "spouse",
            "num_household",
            "income_per_household",
            "months_since_enrollment",
            "deals_purchases_ratio",
        ]
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.pipeline = None
        self.preprocessor = None

    def fit(self, df: pd.DataFrame):
        """Train the clustering pipeline."""
        df = self.preprocess(df)
        X = df[["id"] + self.feature_columns].copy()

        categorical_features = X.select_dtypes(include="object").columns.tolist()
        numerical_features = X.drop(columns=["id"]).select_dtypes(include="number").columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([("scaler", StandardScaler())]),
                    numerical_features,
                ),
                (
                    "cat",
                    Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                    categorical_features,
                ),
            ]
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("pca", PCA(n_components=self.n_components)),
                ("kmeans", KMeans(n_clusters=self.n_clusters, random_state=42)),
            ]
        )

        self.pipeline.fit(X.drop(columns=["id"]))

        # some prints
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Number of components: {self.n_components}")
        print(f"Variance explained: {self.pipeline['pca'].explained_variance_ }")
        inertia = self.pipeline["kmeans"].inertia_
        print(f"Inertia: {inertia}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict clusters for new data."""
        assert self.is_fitted(), "Pipeline is not fitted."
        df = self.preprocess(df)
        df["cluster"] = self.pipeline.predict(df[self.feature_columns])
        self.df = df
        return df

    def dump_pipeline(self, version: Optional[str] = "v1"):
        """Save the trained pipeline."""
        assert self.is_fitted(), "Pipeline is not fitted."
        joblib.dump(self.pipeline, f"models/model_{version}.pkl")

    def load_pipeline(self, version: Optional[str] = "v1"):
        """Load a previously saved pipeline."""
        try:
            self.pipeline = joblib.load(f"models/model_{version}.pkl")
        except FileNotFoundError:
            raise FileNotFoundError(f"Pipeline {version} not found. Train the pipeline first.")

    def is_fitted(self) -> bool:
        """Check if the pipeline is fitted."""
        return self.pipeline is not None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to the dataset."""
        for column in self.outliers_columns:
            df = self.drop_outliers(df, column)
        df = self.impute_missing(df)
        return self.feature_engineering(df)

    @staticmethod
    def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for the dataset."""
        current_year = datetime.today().year
        df["age"] = current_year - df["year_birth"]

        df["num_children"] = df["kidhome"] + df["teenhome"]
        df["spouse"] = (df["marital_status"] == "married").astype(int)
        df["num_household"] = df["num_children"] + df["spouse"] + 1
        df["income_per_household"] = df["income"] / df["num_household"]

        purchase_cols = [
            "mnt_wines",
            "mnt_fruits",
            "mnt_meat_products",
            "mnt_fish_products",
            "mnt_sweet_products",
            "mnt_gold_prods",
        ]
        df["total_purchases"] = df[purchase_cols].sum(axis=1)

        # knowing that the amount of purchases are for the past 2 years, let's have the month average of products
        df["mnt_month_avg_purchases"] = df["total_purchases"] / 24
        df["mnt_month_avg_wines"] = df["mnt_wines"] / 24
        df["mnt_month_avg_fruits"] = df["mnt_fruits"] / 24
        df["mnt_month_avg_meat_products"] = df["mnt_meat_products"] / 24
        df["mnt_month_avg_fish_products"] = df["mnt_fish_products"] / 24
        df["mnt_month_avg_sweet_products"] = df["mnt_sweet_products"] / 24
        df["mnt_month_avg_gold_prods"] = df["mnt_gold_prods"] / 24

        df["mnt_month_avg_purchases"] = df["total_purchases"] / 24

        df["monthly_income"] = df["income"] / 12
        df["dt_customer"] = pd.to_datetime(df["dt_customer"])
        df["months_since_enrollment"] = (datetime.today() - df["dt_customer"]).dt.days / 30
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 40, 50, 60, 100],
            labels=["0-30", "30-40", "40-50", "50-60", "60+"],
        )

        df["num_accepted_campaigns"] = df[
            [
                "accepted_cmp1",
                "accepted_cmp2",
                "accepted_cmp3",
                "accepted_cmp4",
                "accepted_cmp5",
                "response",
            ]
        ].sum(axis=1)

        df["preferred_channel"] = np.argmax(
            df[
                [
                    "num_web_purchases",
                    "num_catalog_purchases",
                    "num_store_purchases",
                ]
            ].values,
            axis=1,
        )
        df["preferred_channel"] = df["preferred_channel"].map({0: "web", 1: "catalog", 2: "store"})
        df["deals_purchases_ratio"] = df["num_deals_purchases"] / df["total_purchases"].replace(0, np.nan)

        return df

    @staticmethod
    def impute_missing(df: pd.DataFrame, input_columns: list = ["income"]) -> pd.DataFrame:
        """Impute missing values using KNNImputer."""
        X = df.drop(
            columns=[
                "id",
                "response",
                "id",
                "dt_customer",
                "education",
                "marital_status",
            ]
        )

        try:
            imputer = joblib.load("models/imputer.pkl")
        except FileNotFoundError:
            model_path = "models/imputer.pkl"

            # Ensure the directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            imputer = KNNImputer(n_neighbors=5)
            X = df.select_dtypes(include=["number"]).drop(columns=["id", "response"])
            imputer.fit(X)
            joblib.dump(imputer, model_path)

        X_imputed = imputer.transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
        for column in input_columns:
            df[column] = X_imputed[column].tolist()
        return df

    @staticmethod
    def drop_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers using the IQR method."""
        iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
        lower_bound = df[column].quantile(0.25) - (1.5 * iqr)
        upper_bound = df[column].quantile(0.75) + (1.5 * iqr)
        return df[((df[column] > lower_bound) & (df[column] < upper_bound)) | (df[column].isna())]


if __name__ == "__main__":
    df = pd.read_csv("data/etl_data.csv")
    model = ModelProcess(n_components=10, n_clusters=5)
    model.fit(df)
    model.dump_pipeline(version="v1")
    model.load_pipeline(version="v1")
