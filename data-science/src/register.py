
import argparse
import json
from pathlib import Path
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_info_output_path", type=str, required=True)

    args = parser.parse_args()

    mlflow.start_run()

    print(f"Registering model '{args.model_name}' from path: {args.model_path}")

    # Register model via MLflow (AzureML integration)
    result = mlflow.register_model(
        model_uri=args.model_path,
        name=args.model_name
    )

    # Write model metadata to pipeline output
    output_dir = Path(args.model_info_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = {
        "model_name": args.model_name,
        "model_version": result.version,
        "model_uri": args.model_path
    }

    with open(output_dir / "model_info.json", "w") as f:
        json.dump(model_info, f)

    print(f"Model registered successfully: {model_info}")

    mlflow.end_run()

if __name__ == "__main__":
    main()
