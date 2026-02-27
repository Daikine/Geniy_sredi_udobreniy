"""
train_nn.py
-----------

This script trains a simple multi-layer perceptron (MLP) regression model
for time-series forecasting of product demand.  For each product in the
input CSV file, an MLPRegressor is trained to predict the next day's
sales given a sequence of past `window` days.  The trained models and
normalization parameters are saved to disk in a single pickle file.

Unlike deep learning libraries such as PyTorch or TensorFlow, this script
uses scikit-learn's MLPRegressor, which is lightweight and does not
require GPU support.  Although the resulting network is shallow,
it still captures non-linear relationships in the time series.

Usage:
    python train_nn.py --data sales_data.csv --window 7 --output models.pkl
"""

from __future__ import annotations

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def build_time_series_dataset(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Transform a 1D time series into a supervised learning dataset.

    Each sample consists of `window` consecutive observations from the series,
    and the target is the value immediately following the window.

    Parameters
    ----------
    series : np.ndarray
        One-dimensional array of time series values.
    window : int
        Number of past observations used to predict the next value.

    Returns
    -------
    X : np.ndarray of shape (n_samples, window)
        Matrix of input sequences.
    y : np.ndarray of shape (n_samples,)
        Vector of target values.
    """
    n_samples = len(series) - window
    X = np.zeros((n_samples, window), dtype=float)
    y = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        X[i] = series[i : i + window]
        y[i] = series[i + window]
    return X, y


def train_product_model(
    series: np.ndarray,
    window: int = 7,
    hidden_layer_sizes: tuple[int, ...] = (50, 25),
    max_iter: int = 500,
    random_state: int | None = None,
) -> dict:
    """Train an MLP model for a single product time series.

    The function splits the series into training and test sets, scales the
    inputs, trains an MLPRegressor, and returns the trained model along with
    scaler and evaluation metrics.

    Parameters
    ----------
    series : np.ndarray
        1D array of sales values for a product.
    window : int
        Sequence length used as input to the model.
    hidden_layer_sizes : tuple[int, ...]
        Architecture of the MLPRegressor (hidden layer sizes).
    max_iter : int
        Maximum number of training iterations.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    info : dict
        Dictionary containing the trained model, scaler, and MAE on the
        held-out test set.
    """
    X, y = build_time_series_dataset(series, window)
    # Split data into training and test sets (80% train / 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Initialize and train MLPRegressor
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
    )
    mlp.fit(X_train_scaled, y_train)
    # Evaluate on test set
    y_pred = mlp.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    return {
        "model": mlp,
        "scaler": scaler,
        "mae": mae,
    }


def train_models_for_all_products(
    df: pd.DataFrame,
    window: int = 7,
    hidden_layer_sizes: tuple[int, ...] = (50, 25),
    max_iter: int = 500,
    random_state: int | None = None,
) -> dict[str, dict]:
    """Train MLP models for each product in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the sales data with columns 'product' and 'sales'.
    window : int
        Sequence length used as input to the model.
    hidden_layer_sizes : tuple[int, ...]
        Architecture of the MLPRegressor.
    max_iter : int
        Maximum number of iterations for training.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    models_info : dict[str, dict]
        Mapping from product name to a dictionary with the trained model,
        scaler and MAE.
    """
    models_info: dict[str, dict] = {}
    for product, group in df.groupby("product"):
        series = group.sort_values("date")["sales"].values.astype(float)
        info = train_product_model(
            series,
            window=window,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
        )
        models_info[product] = info
    return models_info


def save_models(models_info: dict[str, dict], output_path: str) -> None:
    """Serialize the trained models and associated scalers to a pickle file."""
    with open(output_path, "wb") as f:
        pickle.dump(models_info, f)


def main():
    parser = argparse.ArgumentParser(description="Train MLP models for sales forecasting.")
    parser.add_argument(
        "--data",
        type=str,
        default="sales_data.csv",
        help="Path to the CSV file containing sales data",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Sequence length (number of past days) used to predict the next day",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models.pkl",
        help="Path to output pickle file for saving trained models",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.data, parse_dates=["date"])
    models_info = train_models_for_all_products(df, window=args.window, random_state=42)
    save_models(models_info, args.output)
    # Print MAE for each product
    for product, info in models_info.items():
        print(f"Trained MLP for {product} — MAE: {info['mae']:.3f}")
    print(f"Models saved to {args.output}")


if __name__ == "__main__":
    main()