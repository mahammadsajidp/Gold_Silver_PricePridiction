import pandas as pd
import numpy as np
import streamlit as st
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.fillna(method="ffill")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "")
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


@st.cache_data
def find_target_column(df: pd.DataFrame, target: str) -> str:
    if target == "gold":
        candidates = [c for c in df.columns if "gold" in c]
    else:
        candidates = [c for c in df.columns if "silver" in c]

    if not candidates:
        raise ValueError(f"No column with '{target}' found in dataset.")

    return candidates[0]


@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)

    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    metrics = {
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

    return metrics


def prepare_features(df: pd.DataFrame, target_col: str):
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col]).select_dtypes(include=["number"])
    y = df[target_col]
    return X, y


def main():
    st.set_page_config(page_title="Gold & Silver Price Predictor", layout="wide")
    st.title("Gold & Silver Price Predictor")
    st.markdown(
        "Use historical data to train a model and predict future gold or silver prices."
    )

    dataset_path = "Gold-Silver-Data.csv"
    try:
        df = load_data(dataset_path)
    except FileNotFoundError:
        st.error(f"Dataset not found: {dataset_path}. Please upload it to the project root.")
        return

    with st.expander("Dataset Preview"):
        st.dataframe(df.head())

    target_option = st.selectbox("Select target price", ["gold", "silver"])
    target_col = find_target_column(df, target_option)
    st.write(f"### Target column: `{target_col}`")

    X, y = prepare_features(df, target_col)
    st.write(f"Training with {X.shape[1]} numeric features and {len(y)} rows.")

    metrics = train_model(X, y)
    best_model = metrics["lgbm"]["model"]

    st.subheader("Model Evaluation")
    st.write(
        {
            "Linear Regression": {
                "R2": metrics["baseline"]["r2"],
                "RMSE": metrics["baseline"]["rmse"],
            },
            "LightGBM": {
                "R2": metrics["lgbm"]["r2"],
                "RMSE": metrics["lgbm"]["rmse"],
            },
        }
    )

    st.subheader("Predict New Price")
    st.write("Enter feature values below to get a predicted price.")

    input_data = {}
    for feature in X.columns:
        value = st.number_input(feature, value=float(X[feature].median()))
        input_data[feature] = [value]

    if st.button("Predict"):
        new_df = pd.DataFrame(input_data)
        prediction = best_model.predict(new_df)[0]
        st.success(f"Predicted {target_option.title()} Price: {prediction:.4f}")

    with st.expander("Feature correlation matrix"):
        corr = X.corr()
        st.dataframe(corr)


if __name__ == "__main__":
    main()
