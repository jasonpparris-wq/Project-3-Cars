import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    # Accept BOTH flags (some templates use --data, your logs showed --raw_data)
    parser.add_argument("--raw_data", type=str, required=False)
    parser.add_argument("--data", type=str, required=False)

    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    raw_path = args.raw_data or args.data
    if not raw_path:
        raise ValueError("You must provide --raw_data or --data")

    df = pd.read_csv(raw_path)

    # Basic cleanup aligned to your columns
    # ['Segment','Kilometers_Driven','Mileage','Engine','Power','Seats','price']
    # Segment is categorical
    if "Segment" in df.columns:
        df["Segment"] = df["Segment"].astype(str).fillna("Unknown")

    # Coerce numerics
    for col in ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", "price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price"])

    # Simple fill for remaining numeric NA
    for col in ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Output folders are provided by AzureML; still safe to ensure they exist
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    train_path = os.path.join(args.train_data, "train.csv")
    test_path = os.path.join(args.test_data, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train: {train_path} ({len(train_df)} rows)")
    print(f"Saved test : {test_path} ({len(test_df)} rows)")

if __name__ == "__main__":
    main()