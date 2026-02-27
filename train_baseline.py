"""
train_baseline.py
------------------

This script provides a simple baseline model for demand forecasting.
It reads a CSV file containing sales data, splits the data into training
and testing sets by date, and computes a naïve forecast using the
moving average of the last `window` days.  The mean absolute error (MAE)
of the predictions on the test set is printed to the console.

Usage:
    python train_baseline.py --data sales_data.csv --window 7

The baseline is useful as a point of comparison for more sophisticated
models; a neural network should achieve a lower MAE on the test data.
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


def moving_average_forecast(series: np.ndarray, window: int, horizon: int) -> np.ndarray:
    """Compute a moving average forecast.

    For each forecast step, the mean of the last `window` observations is used
    as the prediction. This function iteratively updates the window with
    forecasted values for multi-step forecasts.

    Parameters
    ----------
    series : np.ndarray
        The historical time series values (1D array).
    window : int
        Number of past observations to average.
    horizon : int
        Number of steps to forecast.

    Returns
    -------
    preds : np.ndarray
        Forecasted values for the next `horizon` time steps.
    """
    history = list(series[-window:])  # last 'window' observations
    preds = []
    for _ in range(horizon):
        pred = np.mean(history[-window:])
        preds.append(pred)
        history.append(pred)
    return np.array(preds)


def evaluate_baseline(df: pd.DataFrame, window: int = 7, horizon: int = 1) -> float:
    """Evaluate the baseline model on the provided dataset.

    The dataset must contain the columns 'date', 'product', and 'sales'. The
    evaluation splits the time series per product into training and test
    segments (80% train, 20% test).  For each product, a moving average
    forecast is produced for the test period with a rolling window of
    length `window`.  The MAE across all products is returned.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'date', 'product' and 'sales'.
    window : int
        Number of past observations used for the moving average.
    horizon : int
        Forecast horizon; for the baseline evaluation we set horizon=1 and
        forecast each point individually.

    Returns
    -------
    mae : float
        Mean absolute error across all test points.
    """
    errors = []
    df_sorted = df.sort_values(by=["product", "date"])
    for product, group in df_sorted.groupby("product"):
        series = group["sales"].values.astype(float)
        # Split 80% train, 20% test
        split_idx = int(len(series) * 0.8)
        train, test = series[:split_idx], series[split_idx:]
        # For each test point, generate one-step forecast
        preds = []
        for i in range(len(test)):
            # Use last `window` points from the combined train and previous test predictions
            history = np.concatenate([train, test[:i]])
            # If not enough points, use whatever is available
            window_data = history[-window:] if len(history) >= window else history
            pred = np.mean(window_data)
            preds.append(pred)
        mae = mean_absolute_error(test, preds)
        errors.append(mae)
    return float(np.mean(errors))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a moving-average baseline model.")
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
        help="Window size (number of past days) for moving average",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.data, parse_dates=["date"])
    mae = evaluate_baseline(df, window=args.window)
    print(f"Baseline moving-average MAE (window={args.window}): {mae:.3f}")


if __name__ == "__main__":
    main()