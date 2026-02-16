
# Required imports for training
import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def select_first_file(path: str) -> str:
    """If path is a folder, return the first file inside; otherwise return the file path."""
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if not f.startswith(".")]
        if not files:
            raise FileNotFoundError(f"No files found in directory: {path}")
        return os.path.join(path, files[0])
    return path


def main():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train dataset (file or folder)")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset (file or folder)")
    parser.add_argument("--model_output", type=str, required=True, help="Output directory for MLflow model")
    parser.add_argument("--n_estimators", type=int, default=100, help="The number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=None, help="The maximum depth of the tree")

    args = parser.parse_args()

    # Start MLflow run INSIDE main (important for pipeline execution)
    mlflow.start_run()

    # Load datasets (handles uri_folder mounts)
    train_path = select_first_file(args.train_data)
    test_path = select_first_file(args.test_data)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Split into features and target
    y_train = train_df["price"]
    X_train = train_df.drop(columns=["price"])
    y_test = test_df["price"]
    X_test = test_df.drop(columns=["price"])

    # Ensure all features are numeric (robust to string values like "1197 CC", etc.)
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Log params
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Predict + metric
    yhat_test = model.predict(X_test)
    mse = mean_squared_error(y_test, yhat_test)
    print(f"Mean Squared Error on test set: {mse:.4f}")
    mlflow.log_metric("MSE", float(mse))  # must match sweep primary_metric EXACTLY

    # Save MLflow model DIRECTLY into args.model_output (must contain MLmodel at root)
    os.makedirs(args.model_output, exist_ok=True)
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

    mlflow.end_run()


if __name__ == "__main__":
    main()
