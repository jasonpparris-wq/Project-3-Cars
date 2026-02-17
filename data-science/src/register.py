# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os
import json


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name under which model will be registered")
    parser.add_argument("--model_path", type=str, help="Model directory")
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f"Arguments: {args}")
    return args


def main(args):
    """Loads the best-trained model from the sweep job and registers it"""

    print(f"Registering model: {args.model_name}")

    # FIXED: all MLflow operations must run inside an active run context.
    # Using mlflow.active_run() outside a `with` block returns None in AzureML
    # pipeline steps and causes an AttributeError on .info.run_id.
    with mlflow.start_run():
        # Load model saved by the sweep trial
        model = mlflow.sklearn.load_model(args.model_path)

        # Log the model as an artifact of this registration run
        mlflow.sklearn.log_model(model, args.model_name)

        # Retrieve the run_id from the ACTIVE run (not a stale reference)
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{args.model_name}"

        # Register in the MLflow Model Registry
        mlflow_model = mlflow.register_model(model_uri, args.model_name)
        model_version = mlflow_model.version
        print(f"Registered model '{args.model_name}' as version {model_version}")

        # Write model info JSON for downstream steps / audit trail
        os.makedirs(args.model_info_output_path, exist_ok=True)
        model_info = {"id": f"{args.model_name}:{model_version}"}
        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as of:
            json.dump(model_info, of)
        print(f"Model info written to: {output_path}")


if __name__ == "__main__":
    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}",
    ]
    for line in lines:
        print(line)

    main(args)
