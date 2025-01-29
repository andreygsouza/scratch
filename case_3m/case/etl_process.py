import os
import re

import pandas as pd


class ETLProcess:
    """
    ETL process for the Marketing campaign analysis.
    """

    def __init__(self, data_path: str):
        """
        :data_path: str, path to the data
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist")

        self.data_path = data_path
        self.df = None

    def process(self):
        """
        Wrapper function to process the data
        :returns: pd.DataFrame data processed
        """
        print("Start processing data")
        df = self.read_data()
        print(f"Read data with shape {df.shape}")
        df.columns = [self.camel_to_snake(col) for col in df.columns]
        df = self.convert_to_date(df, ["dt_customer"])
        # convert the string columns to lowercase
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.lower()
        print("Fix marital status")
        df = self.fix_marital_status(df)
        print("Deduplicate rows")
        df = self.deduplicated_rows(df)
        # drop some columns
        df = df.drop(columns=["z_cost_contact", "z_revenue"])

        # Save the processed data
        df.to_csv("data/etl_data.csv", index=False)

        self.df = df
        return df.copy()

    def read_data(self) -> pd.DataFrame:
        """
        Read the data from the excel file
        """
        df = pd.read_excel(self.data_path, sheet_name="data")
        return df

    @staticmethod
    def camel_to_snake(name: str) -> str:
        """
        Convert a string from camelCase to snake_case
        :name: str
        :returns: str
        """
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        # replace double underscores with single
        name = name.replace("__", "_")
        return name

    @staticmethod
    def convert_to_date(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
        """
        Try to convert the date columns to datetime
        :df: pd.DataFrame
        :returns: pd.DataFrame
        """
        for column in date_columns:
            df[column] = pd.to_datetime(df[column], errors="raise")
        return df

    @staticmethod
    def fix_marital_status(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix the marital status column
        :df: pd.DataFrame
        :returns: pd.DataFrame
        """
        df["marital_status"] = (
            df["marital_status"].replace(["absurd", "yolo", "alone"], "single").replace(["together"], "married")
        )
        expected_values = ["single", "married", "divorced", "widow"]
        assert df["marital_status"].isin(expected_values).all(), f"Marital status values are not in {expected_values}"
        return df

    @staticmethod
    def deduplicated_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicated rows from a dataframe
        :df: pd.DataFrame
        :returns: pd.DataFrame
        """
        # get the columns
        columns = df.drop(columns=["id", "response"]).columns.to_list()
        # get the duplicated rows
        df_dup = df[df[columns].duplicated(keep=False)]
        # get the non-duplicated rows
        df_non_dup = df[~df.index.isin(df_dup.index)]
        # sort the duplicated rows
        df_dup = df_dup.sort_values(by=columns)
        # get the target sum and count to check for inconsistency on the duplicated rows
        df_dup["target_sum"] = df_dup.groupby(columns)["response"].transform("sum")
        df_dup["target_count"] = df_dup.groupby(columns)["response"].transform("count")
        # check for inconsistency
        mask = df_dup["target_sum"] == df_dup["target_count"]
        mask |= df_dup["target_sum"] == 0

        # filter the inconsistent duplicated rows and dedup
        df_dedup = df_dup[mask].drop_duplicates(subset=columns, keep="first")
        df_inconsistent = df_dup[~mask]
        print(f"Found {df_inconsistent.shape[0]} inconsistent duplicated rows")
        print(df_dedup.head(10))

        # df_dedup = df.drop(columns=["target_sum", "target_count"])
        # drop target_sum and target_count columns
        df_dedup = df_dedup.drop(columns=["target_sum", "target_count"])
        return pd.concat([df_non_dup, df_dedup], ignore_index=True)


if __name__ == "__main__":
    data_path = "../data/marketing_campaign_wines.xlsx"
    etl = ETLProcess(data_path=data_path)
    df = etl.process()
