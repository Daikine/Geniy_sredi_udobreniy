"""
generate_data.py
------------------

This script generates synthetic sales data for a small assortment of products.
The data simulate seasonality (weekly and annual patterns), trend, and
random noise.  It produces a CSV file that can be used to train and
evaluate forecasting models.

Usage:
    python generate_data.py --output data/sales_data.csv --num-days 1000

The CSV file will have the following columns:

    date: ISO formatted date string (YYYY-MM-DD)
    product: Name of the product (e.g., "Product A")
    price: Simulated sale price (constant with small random fluctuations)
    sales: Number of units sold on that date

The script is deterministic by default; you can set a random seed with
the `--seed` argument to reproduce the same data.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sales_series(
    start_date: datetime,
    num_days: int,
    base_level: float,
    trend: float,
    seasonal_amp: float,
    noise_std: float,
    weekly_pattern: list[float],
    random_state: np.random.Generator,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Generate a synthetic sales time series for one product.

    Parameters
    ----------
    start_date : datetime
        The first date in the series.
    num_days : int
        Number of days to simulate.
    base_level : float
        Base number of units sold.
    trend : float
        Daily trend component (units per day).
    seasonal_amp : float
        Amplitude of the annual seasonal component.
    noise_std : float
        Standard deviation of the Gaussian noise added to the sales.
    weekly_pattern : list[float]
        Multipliers for each day of the week (length 7). For example,
        values >1.0 indicate higher sales on those days, values <1.0 indicate
        lower sales.
    random_state : np.random.Generator
        Random number generator instance.

    Returns
    -------
    dates : pd.DatetimeIndex
        Array of dates of length `num_days`.
    price_series : np.ndarray
        Simulated prices for the product (with small random fluctuations).
    sales_series : np.ndarray
        Simulated daily sales counts.
    """
    dates = pd.date_range(start=start_date, periods=num_days, freq="D")

    # Annual seasonality using a sine wave (one cycle per year).
    days_of_year = np.array([(d - datetime(d.year, 1, 1)).days for d in dates])
    annual_seasonality = seasonal_amp * np.sin(2 * np.pi * days_of_year / 365.25)

    # Weekly pattern: repeating multipliers for each day of the week.
    weekly_mult = np.array([weekly_pattern[d.weekday()] for d in dates])

    # Trend component: linear increase over time.
    trend_component = trend * np.arange(num_days)

    # Base level and combined seasonality/trend.
    true_sales = base_level + trend_component + annual_seasonality
    true_sales = true_sales * weekly_mult

    # Add random noise.
    noise = random_state.normal(loc=0.0, scale=noise_std, size=num_days)

    sales = true_sales + noise
    # Ensure sales are non-negative and round to integers.
    sales = np.maximum(sales, 0)
    sales = np.round(sales).astype(int)

    # Simulate prices: base price with small daily fluctuations.
    base_price = 10.0  # arbitrary base price
    price_fluctuation = random_state.normal(loc=0.0, scale=0.1, size=num_days)
    price_series = base_price + price_fluctuation
    price_series = np.round(price_series, 2)

    return dates, price_series, sales


def create_dataset(
    output_path: str,
    num_days: int = 1000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a multi-product synthetic sales dataset and save to CSV.

    The dataset includes several products with different seasonality and trend
    characteristics. Data for each product are concatenated together with
    identical date ranges.

    Parameters
    ----------
    output_path : str
        Path to save the generated CSV file.
    num_days : int, optional
        Number of days to simulate for each product, by default 1000.
    seed : int | None, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the generated data.
    """
    rng = np.random.default_rng(seed)
    start_date = datetime.now().date() - timedelta(days=num_days)

    products = [
        {
            "name": "Product A",
            "base_level": 50,
            "trend": 0.02,
            "seasonal_amp": 15,
            "noise_std": 5,
            "weekly_pattern": [1.0, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3],
        },
        {
            "name": "Product B",
            "base_level": 30,
            "trend": -0.01,
            "seasonal_amp": 10,
            "noise_std": 3,
            "weekly_pattern": [0.8, 0.85, 0.9, 1.0, 1.2, 1.5, 1.6],
        },
        {
            "name": "Product C",
            "base_level": 70,
            "trend": 0.05,
            "seasonal_amp": 20,
            "noise_std": 8,
            "weekly_pattern": [1.2, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8],
        },
    ]

    all_records = []
    for product in products:
        dates, prices, sales = generate_sales_series(
            start_date=start_date,
            num_days=num_days,
            base_level=product["base_level"],
            trend=product["trend"],
            seasonal_amp=product["seasonal_amp"],
            noise_std=product["noise_std"],
            weekly_pattern=product["weekly_pattern"],
            random_state=rng,
        )

        product_records = pd.DataFrame(
            {
                "date": dates,
                "product": product["name"],
                "price": prices,
                "sales": sales,
            }
        )
        all_records.append(product_records)

    df = pd.concat(all_records, ignore_index=True)
    # Sort by date and product for neatness.
    df = df.sort_values(by=["product", "date"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sales data.")
    parser.add_argument(
        "--output",
        type=str,
        default="sales_data.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--num-days",
        type=int,
        default=1000,
        help="Number of days to simulate for each product",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    create_dataset(args.output, num_days=args.num_days, seed=args.seed)
    print(f"Synthetic sales data saved to {args.output}")


if __name__ == "__main__":
    main()