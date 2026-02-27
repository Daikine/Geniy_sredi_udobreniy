"""
app.py
------

Graphical interface for demand forecasting.

This script starts a simple Tkinter GUI that allows a user to select a
pre-trained product model or load a custom CSV file containing historical
sales data.  Upon clicking the "Forecast" button, the application
displays a plot of the recent sales history along with the forecast for
the next 14 days.  The forecasting uses a pre-trained MLP model for
built-in products and a naïve moving-average baseline for custom CSVs.

Usage:
    python app.py

Dependencies:
    - pandas
    - numpy
    - matplotlib
    - scikit-learn (for scaler)
    - tkinter (built into the Python standard library)

All dependencies except Tkinter should already be installed in the
execution environment; this script does not require Streamlit or
additional web frameworks.
"""

from __future__ import annotations

import os
import pickle
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib

# Use Agg backend for embedding in Tkinter
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402


DATA_FILE = "sales_data.csv"
MODELS_FILE = "models.pkl"
FORECAST_HORIZON = 14
WINDOW_SIZE = 7  # Should match the window used in training


class ForecastApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Demand Forecasting App")
        master.geometry("800x600")

        # Load dataset and models (if available)
        self.df: pd.DataFrame | None = None
        self.models_info: dict[str, dict] | None = None
        self.load_dataset(DATA_FILE)
        self.load_models(MODELS_FILE)

        # GUI Elements
        self.product_var = tk.StringVar(master)
        # Initialize product list from dataframe
        products = sorted(self.df["product"].unique()) if self.df is not None else []
        self.product_var.set(products[0] if products else "")

        tk.Label(master, text="Select product:").pack(pady=5)
        self.product_menu = ttk.Combobox(master, textvariable=self.product_var, values=products)
        self.product_menu.pack()

        # Button to load custom CSV
        tk.Button(master, text="Load CSV", command=self.load_custom_csv).pack(pady=10)
        self.custom_file_label = tk.Label(master, text="No custom file loaded")
        self.custom_file_label.pack()

        # Forecast button
        tk.Button(master, text="Generate Forecast", command=self.generate_forecast).pack(pady=10)

        # Canvas for plot
        self.figure = plt.Figure(figsize=(8, 4))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Data for custom CSV
        self.custom_df: pd.DataFrame | None = None

    def load_dataset(self, path: str) -> None:
        if os.path.exists(path):
            try:
                self.df = pd.read_csv(path, parse_dates=["date"])
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to load dataset {path}: {exc}")
                self.df = None
        else:
            self.df = None

    def load_models(self, path: str) -> None:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.models_info = pickle.load(f)
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to load models {path}: {exc}")
                self.models_info = None
        else:
            self.models_info = None

    def load_custom_csv(self) -> None:
        filepath = filedialog.askopenfilename(
            title="Open CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filepath:
            return
        try:
            df = pd.read_csv(filepath)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to read CSV: {exc}")
            return
        # Ensure the file has at least two columns: date and sales
        cols = df.columns.str.lower().tolist()
        date_col_candidates = [c for c in df.columns if c.lower() in {"date", "ds"}]
        sales_col_candidates = [c for c in df.columns if c.lower() in {"sales", "y", "value"}]
        if not date_col_candidates or not sales_col_candidates:
            messagebox.showerror(
                "Invalid format",
                "CSV must contain date and sales columns (e.g., 'date' and 'sales').",
            )
            return
        date_col = date_col_candidates[0]
        sales_col = sales_col_candidates[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to parse dates: {exc}")
            return
        df = df.sort_values(by=date_col).reset_index(drop=True)
        # Rename for consistency
        df = df[[date_col, sales_col]].rename(columns={date_col: "date", sales_col: "sales"})
        self.custom_df = df
        self.custom_file_label.config(text=f"Loaded: {os.path.basename(filepath)}")
        # Reset product selection to none for clarity
        self.product_var.set("")
        self.product_menu.configure(state="disabled")
    
    def reset_to_built_in(self) -> None:
        """Reset state to built-in dataset (disable custom).

        Called when the user selects a product after loading a custom file.
        """
        self.custom_df = None
        self.custom_file_label.config(text="No custom file loaded")
        self.product_menu.configure(state="readonly")

    def generate_forecast(self) -> None:
        """Generate and display the forecast based on the current selection."""
        # Clear the previous plot
        self.ax.clear()
        # If custom data is loaded, use baseline forecast
        if self.custom_df is not None:
            self._forecast_custom_data()
        else:
            product = self.product_var.get()
            if not product:
                messagebox.showinfo("No product", "Please select a product or load a CSV file.")
                return
            self._forecast_built_in(product)
        self.canvas.draw()

    def _forecast_built_in(self, product: str) -> None:
        assert self.df is not None and self.models_info is not None, "Built-in data or models not loaded"
        # Extract series for the product
        group = self.df[self.df["product"] == product].sort_values("date")
        sales_series = group["sales"].values.astype(float)
        dates = group["date"].values
        # Retrieve model and scaler
        info = self.models_info.get(product)
        if info is None:
            messagebox.showerror("Model missing", f"No model found for {product}")
            return
        model = info["model"]
        scaler = info["scaler"]
        window = WINDOW_SIZE
        history = list(sales_series)
        preds = []
        for _ in range(FORECAST_HORIZON):
            # Prepare input sequence
            x_input = np.array(history[-window:]).reshape(1, -1)
            x_scaled = scaler.transform(x_input)
            pred = model.predict(x_scaled)[0]
            preds.append(pred)
            history.append(pred)
        # Build forecast dates
        last_date = dates[-1]
        forecast_dates = [last_date + np.timedelta64(i + 1, "D") for i in range(FORECAST_HORIZON)]
        # Plot last 60 days of actual data and forecast
        history_days = min(len(sales_series), 60)
        self.ax.plot(dates[-history_days:], sales_series[-history_days:], label="Actual sales")
        self.ax.plot(
            forecast_dates,
            preds,
            label="Forecast (MLP)",
            linestyle="--",
            marker="o",
            color="tab:red",
        )
        self.ax.set_title(f"Demand Forecast for {product}")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Sales")
        self.ax.legend()
        self.ax.grid(True)
        self.figure.autofmt_xdate()

    def _forecast_custom_data(self) -> None:
        assert self.custom_df is not None
        df = self.custom_df
        series = df["sales"].values.astype(float)
        dates = df["date"].values
        # Use moving-average baseline with window=WINDOW_SIZE
        window = WINDOW_SIZE if len(series) > WINDOW_SIZE else max(1, len(series) // 2)
        history = list(series)
        preds = []
        for _ in range(FORECAST_HORIZON):
            window_data = history[-window:]
            pred = float(np.mean(window_data))
            preds.append(pred)
            history.append(pred)
        last_date = dates[-1]
        forecast_dates = [last_date + np.timedelta64(i + 1, "D") for i in range(FORECAST_HORIZON)]
        # Plot entire history and forecast
        self.ax.plot(dates, series, label="Actual sales")
        self.ax.plot(
            forecast_dates,
            preds,
            label="Forecast (Moving Avg)",
            linestyle="--",
            marker="o",
            color="tab:green",
        )
        self.ax.set_title("Demand Forecast (Custom Data)")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Sales")
        self.ax.legend()
        self.ax.grid(True)
        self.figure.autofmt_xdate()


def main():
    root = tk.Tk()
    app = ForecastApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()