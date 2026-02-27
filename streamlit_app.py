import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import matplotlib.pyplot as plt

DATA_FILE = "sales_data.csv"
MODELS_FILE = "models.pkl"
WINDOW_SIZE = 7
FORECAST_HORIZON = 14

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_resource
def load_models(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def forecast_with_model(model, scaler, series: np.ndarray, window: int = WINDOW_SIZE) -> np.ndarray:
    history = list(series)
    preds = []
    for _ in range(FORECAST_HORIZON):
        x_input = np.array(history[-window:]).reshape(1, -1)
        x_scaled = scaler.transform(x_input)
        pred = float(model.predict(x_scaled)[0])
        preds.append(pred)
        history.append(pred)
    return np.array(preds)

def forecast_moving_average(series: np.ndarray, window: int) -> np.ndarray:
    history = list(series)
    preds = []
    for _ in range(FORECAST_HORIZON):
        window_data = history[-window:] if len(history) >= window else history
        pred = float(np.mean(window_data))
        preds.append(pred)
        history.append(pred)
    return np.array(preds)

def parse_custom_csv(uploaded_file) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Не удалось прочитать CSV: {exc}")
        return None
    cols_lower = [c.lower() for c in df.columns]
    date_candidates = [c for c in df.columns if c.lower() in {"date", "ds"}]
    sales_candidates = [c for c in df.columns if c.lower() in {"sales", "y", "value"}]
    if not date_candidates or not sales_candidates:
        st.error("CSV должен содержать столбцы с датой (date или ds) и продажами (sales, y или value)")
        return None
    date_col = date_candidates[0]
    sales_col = sales_candidates[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as exc:
        st.error(f"Не удалось разобрать даты: {exc}")
        return None
    df = df.sort_values(by=date_col).reset_index(drop=True)
    return df[[date_col, sales_col]].rename(columns={date_col: "date", sales_col: "sales"})

def main():
    st.set_page_config(page_title="Demand Forecasting App")
    st.title("Demand Forecasting")

    st.sidebar.header("Настройки")
    use_custom = st.sidebar.checkbox("Использовать свой CSV")

    if use_custom:
        uploaded_file = st.sidebar.file_uploader(
            "Загрузите CSV с историей продаж", type=["csv"], accept_multiple_files=False
        )
        if uploaded_file:
            df_custom = parse_custom_csv(uploaded_file)
            if df_custom is not None:
                st.write("### Загруженные данные")
                st.dataframe(df_custom.tail(10))
                window = WINDOW_SIZE if len(df_custom) > WINDOW_SIZE else max(1, len(df_custom) // 2)
                series = df_custom["sales"].values.astype(float)
                preds = forecast_moving_average(series, window)
                last_date = df_custom["date"].iloc[-1]
                forecast_dates = [last_date + timedelta(days=i + 1) for i in range(FORECAST_HORIZON)]
                fig, ax = plt.subplots()
                ax.plot(df_custom["date"], series, label="Фактические продажи")
                ax.plot(forecast_dates, preds, label="Прогноз (скользящее среднее)", linestyle="--", marker="o")
                ax.set_xlabel("Дата")
                ax.set_ylabel("Продажи")
                ax.set_title("Прогноз спроса (пользовательский CSV)")
                ax.legend()
                ax.grid(True)
                fig.autofmt_xdate()
                st.pyplot(fig)
        else:
            st.info("Выберите файл CSV в панели слева")
    else:
        data = load_data(DATA_FILE)
        models = load_models(MODELS_FILE)
        products = sorted(data["product"].unique())
        product = st.sidebar.selectbox("Выберите товар", products)
        if st.sidebar.button("Рассчитать прогноз"):
            group = data[data["product"] == product].sort_values("date")
            series = group["sales"].values.astype(float)
            model = models[product]["model"]
            scaler = models[product]["scaler"]
            preds = forecast_with_model(model, scaler, series)
            last_date = group["date"].iloc[-1]
            forecast_dates = [last_date + timedelta(days=i + 1) for i in range(FORECAST_HORIZON)]
            fig, ax = plt.subplots()
            history_days = min(len(series), 60)
            ax.plot(group["date"].iloc[-history_days:], series[-history_days:], label="Фактические продажи")
            ax.plot(forecast_dates, preds, label="Прогноз (MLP)", linestyle="--", marker="o")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Продажи")
            ax.set_title(f"Прогноз спроса для {product}")
            ax.legend()
            ax.grid(True)
            fig.autofmt_xdate()
            st.pyplot(fig)
        st.write(
            "### Исторические данные", data[data["product"] == product].tail(10)
        )

if __name__ == "__main__":
    main()