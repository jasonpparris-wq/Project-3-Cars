import argparse
import os
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--model_output", type=str, required=True)
    return parser.parse_args()


def load_split(folder: str, filename: str) -> pd.DataFrame:
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_csv(path)


def main():
    args = parse_args()

    train_df = load_split(args.train_data, "train.csv")
    test_df = load_split(args.test_data, "test.csv")

    target_col = "price"
    if target_col not in train_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in training data columns: "
            f"{train_df.columns.tolist()}"
        )

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    cat_features = [c for c in X_train.columns if X_train[c].dtype == "object"]
    num_features = [c for c in X_train.columns if c not in cat_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # FIXED: use context manager so MLflow run is always cleanly ended,
    # even if an exception is raised mid-training.
    with mlflow.start_run():
        mlflow.log_param("model", "used-cars-random-tree")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        # primary_metric in newpipeline.yml sweep objective is "rmse" — must match exactly
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        os.makedirs(args.model_output, exist_ok=True)
        mlflow.sklearn.save_model(sk_model=pipe, path=args.model_output)

        print(f"Saved MLflow model to: {args.model_output}")
        print(f"rmse={rmse:.4f}, mae={mae:.4f}, r2={r2:.4f}")


if __name__ == "__main__":
    main()
