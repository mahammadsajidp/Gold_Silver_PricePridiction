import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.fillna(method="ffill")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "")
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def find_target_column(df: pd.DataFrame, target: str) -> str:
    if target == "gold":
        candidates = [c for c in df.columns if "gold" in c]
    elif target == "silver":
        candidates = [c for c in df.columns if "silver" in c]
    else:
        raise ValueError("Target must be 'gold' or 'silver'.")

    if not candidates:
        raise ValueError(f"No column with '{target}' found in dataset.")

    return candidates[0]


def prepare_features(df: pd.DataFrame, target_col: str):
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col]).select_dtypes(include=["number"])
    y = df[target_col]

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found after dropping the target.")

    return X, y


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)

    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    results = {
        "baseline": {
            "model": baseline,
            "r2": r2_score(y_test, baseline.predict(X_test)),
            "rmse": np.sqrt(mean_squared_error(y_test, baseline.predict(X_test))),
        },
        "lgbm": {
            "model": model,
            "r2": r2_score(y_test, model.predict(X_test)),
            "rmse": np.sqrt(mean_squared_error(y_test, model.predict(X_test))),
        },
    }

    return results, X_train.columns


def prompt_user_values(feature_names):
    values = {}
    print("Enter values for the following features:")

    for feature in feature_names:
        while True:
            try:
                raw = input(f"  {feature}: ")
                values[feature] = [float(raw)]
                break
            except ValueError:
                print("  Please enter a numeric value.")

    return pd.DataFrame(values)


def save_model(model, target):
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", f"{target}_model.pkl")
    joblib.dump(model, path)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Train and predict gold or silver price models."
    )
    parser.add_argument("--dataset", default="Gold-Silver-Data.csv", help="Path to CSV dataset.")
    parser.add_argument("--target", default="gold", choices=["gold", "silver"], help="Target price to predict.")
    parser.add_argument("--predict", action="store_true", help="Prompt for feature values and predict price.")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model to models/.")
    args = parser.parse_args()

    df = load_data(args.dataset)
    target_col = find_target_column(df, args.target)
    print(f"Using target column: {target_col}")

    X, y = prepare_features(df, target_col)
    train_results, feature_names = train_models(X, y)

    print("\nModel evaluation results:")
    for name, result in train_results.items():
        print(
            f"  {name.title()} - R2: {result['r2']:.4f}, RMSE: {result['rmse']:.4f}"
        )

    best_model = train_results["lgbm"]["model"]

    if args.save_model:
        saved_path = save_model(best_model, args.target)
        print(f"Saved model to {saved_path}")

    if args.predict:
        new_data = prompt_user_values(feature_names)
        prediction = best_model.predict(new_data)
        print(f"\nPredicted {args.target.title()} Price: {prediction[0]:.4f}")


if __name__ == "__main__":
    main()
