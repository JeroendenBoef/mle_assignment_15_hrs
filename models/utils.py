import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from models.constants import CATEGORICAL_COLLUMNS


def read_and_merge_dataset(
    x_filepath: str = "./data/water_pump_set.csv", y_filepath: str = "./data/water_pump_labels.csv"
):
    X = pd.read_csv(x_filepath)
    y = pd.read_csv(y_filepath)

    merged_df = pd.concat([X, y.drop("id", axis=1)], axis=1)

    # Converts dtypes to most suitable (e.g. converting years from string to int64)
    merged_df = merged_df.convert_dtypes()

    # Convert date_recorded to actual dates
    merged_df["date_recorded"] = pd.to_datetime(merged_df["date_recorded"])

    def map_target_label(label):
        # Maps categorical variable to integer
        if label == "non functional":
            return 0
        elif label == "functional needs repair":
            return 1
        else:
            return 2

    merged_df["status_group"] = merged_df["status_group"].apply(map_target_label)

    return merged_df


def encode_numerical(df, column_name):
    categories = df[column_name].value_counts().index.tolist()
    numerical_categories = list(range(len(categories)))
    return df[column_name].replace(categories, numerical_categories).values


def convert_to_unix_timedelta(df: pd.DataFrame):
    current_timestamp = time.time()
    times = pd.to_datetime(df.date_recorded).astype(int) / 10**9
    deltas = [int(current_timestamp) - timestamp for timestamp in times.astype(int)]
    return deltas


def clean_df(df: pd.DataFrame):
    nan_check = df.isna().any()
    nan_columns = [col for col, value in zip(nan_check.index, nan_check) if value == True]
    bool_cols = [col for col, value in zip(nan_columns, df[nan_columns].dtypes) if value == "boolean"]
    for col in bool_cols:
        df[col] = encode_numerical(df, col)
        df[col] = df[col].fillna(2)  # 3rd option for unknown
    df.fillna("", inplace=True)

    for col in [word for word in CATEGORICAL_COLLUMNS if word not in bool_cols]:  # prevent double encoding
        df[col] = encode_numerical(df, col)

    return df


def generate_dataloaders(df: pd.DataFrame, categorical_cols: List[str]):
    train, test = train_test_split(df, test_size=0.2, stratify=df["status_group"])
    val, test = train_test_split(test, test_size=0.5, stratify=test["status_group"])
    feature_cols = df.columns.tolist()[:-1]
    label_col = "status_group"

    train_loader = lgb.Dataset(
        train[feature_cols].values,
        label=train[label_col].values,
        feature_name=feature_cols,
        categorical_feature=categorical_cols,
    )
    val_loader = lgb.Dataset(
        val[feature_cols].values,
        label=val[label_col].values,
        feature_name=feature_cols,
        categorical_feature=categorical_cols,
        reference=train_loader,
    )

    X_test = test[feature_cols].values
    y_test = test[label_col].values

    return train_loader, val_loader, X_test, y_test


def evaluate(model, X_test, y_test):
    logits = model.predict(X_test)
    y_pred = np.argmax(logits, axis=1)

    return classification_report(y_test, y_pred)
