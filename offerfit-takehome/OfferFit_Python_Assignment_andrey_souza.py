"""
Instructions:

- Fill in the methods of the DataModeler class to produce the same printed results
  as in the comments labeled '<Expected Output>' in the second half of the file.
- The DataModeler should predict the 'outcome' from the columns 'amount' and 'transaction date.'
  Your model should ignore the 'customer_id' column.
- For the modeling methods `fit`, `predict` and `model_summary` you can use any appropriate method.
  Try to get 100% accuracy on both training and test, as indicated in the output.
- Please feel free to import any popular libraries of choice for your solution!
- Your solution will be judged on both correctness and code quality.
- Good luck, and have fun!

"""

from __future__ import annotations

import logging
import textwrap

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def is_fitted(estimator):
    """
    Check if the estimator is fitted
    """
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False


class DataModeler:
    def __init__(self, sample_df: pd.DataFrame):
        """
        Initialize the DataModeler as necessary.
        """
        self.sample_df = sample_df

        # initialize the model and data
        self.model = None
        self.train_df = None
        self.train_target = None

        # initialize the feature list
        self.features = ["amount", "transaction_date"]

    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame | None:
        """
        Prepare a dataframe so it contains only the columns to model and having suitable types.
        If the argument is None, work on the training data passed in the constructor.
        """
        df = self.sample_df.copy() if oos_df is None else oos_df.copy()
        if oos_df is None:
            logging.info("Preparing training data")
            self.train_df = self.sample_df.copy()
            self.train_df = self.train_df.set_index("customer_id")[self.features]
            self.train_target = self.sample_df.set_index("customer_id")["outcome"].astype(bool)

            # convert transaction_date to a numeric value
            logging.info("Converting data types")
            self.train_df["transaction_date"] = pd.to_datetime(self.train_df["transaction_date"]).astype(int)
            # keep the NaT values as NaN after conversion
            self.train_df["transaction_date"] = np.where(
                self.sample_df["transaction_date"].isnull(), np.nan, self.train_df["transaction_date"]
            )
            self.train_df["amount"] = self.train_df["amount"].astype(float)

        else:
            logging.info("Preparing out of sample data")
            assert all(
                [col in oos_df.columns for col in self.features]
            ), f"Missing columns in out of sample data: {set(self.features) - set(oos_df.columns)}"
            df = oos_df.copy()
            if "customer_id" in oos_df.columns:
                df = df.set_index("customer_id")
            df = df[self.features]
            df["transaction_date"] = pd.to_datetime(df["transaction_date"]).astype(int)
            df["transaction_date"] = np.where(oos_df["transaction_date"].isnull(), np.nan, df["transaction_date"])
            df["amount"] = df["amount"].astype(float)
            return df

    def impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame | None:
        """
        Fill any missing values with the appropriate mean (average) value.
        If the argument is None, work on the training data passed in the constructor.
        Hint: Watch out for data leakage in your solution.
        """
        if oos_df is None:
            logging.info("Imputing missing values in training data")
            self.train_df["amount"] = self.train_df["amount"].fillna(self.train_df["amount"].mean())
            self.train_df["transaction_date"] = self.train_df["transaction_date"].fillna(
                self.train_df["transaction_date"].mean()
            )
        else:
            logging.info("Imputing missing values in out of sample data")
            oos_df["amount"] = oos_df["amount"].fillna(oos_df["amount"].mean())

            oos_df["transaction_date"] = oos_df["transaction_date"].fillna(oos_df["transaction_date"].mean())
            return oos_df

    def fit(self) -> None:
        """
        Fit the model of your choice on the training data paased in the constructor, assuming it has
        been prepared by the functions prepare_data and impute_missing
        """
        logging.info("Fitting model")
        # self.model = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('classifier', LogisticRegression(random_state=42))
        #
        # ])
        # LogisticRegression with StandardScaler only got 90% accuracy on the training sample
        # use DecisionTreeClassifier instead
        self.model = DecisionTreeClassifier(random_state=42, max_depth=3)
        self.model.fit(self.train_df, self.train_target)
        accuracy = self.model.score(self.train_df, self.train_target)
        logging.info(f"Model accuracy: {accuracy:.2%}")

    def model_summary(self) -> str:
        """
        Create a short summary of the model you have fit.
        """
        # check if the self.model is fitted
        assert is_fitted(self.model), "Model is not fitted. Please fit the model first."

        summary = textwrap.dedent(  # use textwrap to format the string, while preserving the indentation
            f"""
            Model: {self.model.__class__.__name__}
            Number of features: {self.train_df.shape[1]}
            Number of samples: {self.train_df.shape[0]}
            Binary target average: {self.train_target.mean():.2%}
            Accuracy: {self.model.score(self.train_df, self.train_target):.2%}
            """
        )
        return summary

    def predict(self, oos_df: pd.DataFrame = None) -> pd.Series[bool]:
        """
        Make a set of predictions with your model. Assume the data has been prepared by the
        functions prepare_data and impute_missing.
        If the argument is None, work on the training data passed in the constructor.
        """
        # check if the self.model is fitted
        assert is_fitted(self.model), "Model is not fitted. Please fit the model first."
        if oos_df is None:
            logging.info("Predicting on training sample")
            return pd.Series(self.model.predict(self.train_df), index=self.train_df.index)
        else:
            logging.info("Predicting on out of sample data")
            return pd.Series(self.model.predict(oos_df), index=oos_df.index)

    def save(self, path: str) -> None:
        """
        Save the DataModeler so it can be re-used.
        """
        assert is_fitted(self.model), "Model is not fitted. Please fit the model first."
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> DataModeler:
        """
        Reload the DataModeler from the saved state so it can be re-used.
        """
        return joblib.load(path)


#################################################################################
# You should not have to modify the code below this point

transact_train_sample = pd.DataFrame(
    {
        "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
        "transaction_date": [
            "2022-01-01",
            "2022-08-01",
            None,
            "2022-12-01",
            "2022-02-01",
            None,
            "2022-02-01",
            "2022-01-01",
            "2022-11-01",
            "2022-01-01",
        ],
        "outcome": [False, True, True, True, False, False, True, True, True, False],
    }
)


print(f"Training sample:\n{transact_train_sample}\n")

# <Expected Output>
# Training sample:
#    customer_id  amount transaction_date  outcome
# 0           11     1.0       2022-01-01    False
# 1           12     3.0       2022-08-01     True
# 2           13    12.0             None     True
# 3           14     6.0       2022-12-01     True
# 4           15     0.5       2022-02-01    False
# 5           16     0.2             None    False
# 6           17     NaN       2022-02-01     True
# 7           18     5.0       2022-01-01     True
# 8           19     NaN       2022-11-01     True
# 9           20     3.0       2022-01-01    False


print(f"Current dtypes:\n{transact_train_sample.dtypes}\n")

# <Expected Output>
# Current dtypes:
# customer_id           int64
# amount              float64
# transaction_date     object
# outcome                bool
# dtype: object

transactions_modeler = DataModeler(transact_train_sample)

transactions_modeler.prepare_data()

print(f"Changed columns to:\n{transactions_modeler.train_df.dtypes}\n")

# <Expected Output>
# Changed columns to:
# amount              float64
# transaction_date    float64
# dtype: object

transactions_modeler.impute_missing()

print(f"Imputed missing as mean:\n{transactions_modeler.train_df}\n")

# <Expected Output>
# Imputed missing as mean:
#               amount  transaction_date
# customer_id
# 11            1.0000      1.640995e+18
# 12            3.0000      1.659312e+18
# 13           12.0000      1.650845e+18
# 14            6.0000      1.669853e+18
# 15            0.5000      1.643674e+18
# 16            0.2000      1.650845e+18
# 17            3.8375      1.643674e+18
# 18            5.0000      1.640995e+18
# 19            3.8375      1.667261e+18
# 20            3.0000      1.640995e+18


print("Fitting  model")
transactions_modeler.fit()

print(f"Fit model:\n{transactions_modeler.model_summary()}\n")

# <Expected Output>
# Fitting  model
# Fit model:
# <<< ANY SHORT SUMMARY OF THE MODEL YOU CHOSE >>>

in_sample_predictions = transactions_modeler.predict()
print(f"Predicted on training sample: {in_sample_predictions}\n")
print(
    f"Accuracy = {sum(in_sample_predictions ==  [False, True, True, True, False, False, True, True, True, False])/.1}%"
)

# <Expected Output>
# Predicting on training sample [False  True  True  True False False True  True  True False]
# Accuracy = 100.0%

transactions_modeler.save("transact_modeler")
loaded_modeler = DataModeler.load("transact_modeler")

print(f"Loaded DataModeler sample df:\n{loaded_modeler.model_summary()}\n")

# <Expected Output>
# Loaded DataModeler sample df:
# <<< THE SUMMARY OF THE MODEL YOU CHOSE >>>

transact_test_sample = pd.DataFrame(
    {
        "customer_id": [21, 22, 23, 24, 25],
        "amount": [0.5, np.nan, 8, 3, 2],
        "transaction_date": ["2022-02-01", "2022-11-01", "2022-06-01", None, "2022-02-01"],
    }
)

adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)

print(f"Changed columns to:\n{adjusted_test_sample.dtypes}\n")

# <Expected Output>
# Changed columns to:
# amount              float64
# transaction_date    float64
# dtype: object

filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)

print(f"Imputed missing as mean:\n{filled_test_sample}\n")

# <Expected Output>
# Imputed missing as mean:
#              amount  transaction_date
# customer_id
# 21           0.5000      1.643674e+18
# 22           3.8375      1.667261e+18
# 23           8.0000      1.654042e+18
# 24           3.0000      1.650845e+18
# 25           2.0000      1.643674e+18

oos_predictions = transactions_modeler.predict(filled_test_sample)
print(f"Predicted on out of sample data: {oos_predictions}\n")
print(f"Accuracy = {sum(oos_predictions == [False, True, True, False, False])/.05}%")

# <Expected Output>
# Predicted on out of sample data: [False True True False False] ([0 1 1 0 0])
# Accuracy = 100.0%
