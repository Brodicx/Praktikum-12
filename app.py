import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


st.set_page_config(page_title="Jurnal Investigasi Time Series", layout="wide")
sns.set(style="whitegrid")

DATA_PATH = "time series data.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")
    df = df.set_index("date").sort_index()
    product_cols = [c for c in df.columns if c.startswith("ProductP")]
    df["total_products"] = df[product_cols].sum(axis=1)
    return df


def compute_clean_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    date_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = date_range.difference(df.index)
    clean = df.copy()
    clean = clean.interpolate(method="linear").ffill().bfill()
    return clean, missing


def plot_raw_vs_weekly(df: pd.DataFrame, weekly: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df.index, df["total_products"], label="Total Produk", color="tab:blue")
    axes[0].set_title("Total Produk per Hari (data mentah)")
    axes[0].legend()

    axes[1].plot(df.index, df["total_products"], alpha=0.4, label="Harian")
    axes[1].plot(
        weekly.index,
        weekly["total_products"],
        color="tab:orange",
        label="Rata-rata Mingguan",
        linewidth=2,
    )
    axes[1].set_title("Perbandingan Harian vs Rata-rata Mingguan")
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_decompose(series: pd.Series):
    decomp = seasonal_decompose(series, model="additive", period=7)
    fig = decomp.plot()
    fig.set_size_inches(12, 9)
    plt.tight_layout()
    return fig


def plot_rolling(series: pd.Series, window: int):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series, label="Data Asli", alpha=0.5)
    ax.plot(rolling_mean, label=f"Rolling Mean ({window})", color="tab:orange")
    ax.plot(rolling_std, label=f"Rolling Std ({window})", color="tab:green")
    ax.set_title("Rolling Mean & Std di atas Data Asli")
    ax.legend()
    plt.tight_layout()
    return fig


def adf_summary(series: pd.Series):
    result = adfuller(series)
    stat, pvalue, usedlag, nobs, crit, _ = result
    return {
        "ADF Statistic": stat,
        "p-value": pvalue,
        "usedlag": usedlag,
        "nobs": nobs,
        "critical values": crit,
    }


def main():
    st.title("Jurnal Investigasi Time Series")
    st.write("Eksplorasi dataset `time series data.csv` dengan langkah-langkah jurnal.")

    df = load_data(DATA_PATH)
    clean_df, missing_dates = compute_clean_df(df)
    weekly_df = clean_df.resample("W").mean()
    series = clean_df["total_products"]

    st.sidebar.header("Pengaturan")
    rolling_window = st.sidebar.slider("Window Rolling Mean/Std", 5, 30, 12, 1)

    st.subheader("Part 1 — The First Encounter (Setup & Cleaning)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah baris", len(clean_df))
    col2.metric("Gap tanggal hilang", len(missing_dates))
    col3.metric("Rentang tanggal", f"{clean_df.index.min().date()} s.d. {clean_df.index.max().date()}")

    with st.expander("Lihat sampel data", expanded=False):
        st.dataframe(clean_df.head())

    st.subheader("Part 2 — Visual Inspection")
    fig_raw_weekly = plot_raw_vs_weekly(clean_df, weekly_df)
    st.pyplot(fig_raw_weekly)

    st.subheader("Part 3 — Decomposing the Pattern")
    fig_decomp = plot_decompose(series)
    st.pyplot(fig_decomp)

    st.subheader("Part 4 — Statistical Health Check")
    fig_roll = plot_rolling(series, rolling_window)
    st.pyplot(fig_roll)

    st.markdown("**Uji ADF (Augmented Dickey-Fuller)**")
    adf = adf_summary(series)
    st.write(
        pd.DataFrame(
            {
                "ADF Statistic": [adf["ADF Statistic"]],
                "p-value": [adf["p-value"]],
                "usedlag": [adf["usedlag"]],
                "nobs": [adf["nobs"]],
            }
        )
    )
    st.json(adf["critical values"])

    st.caption("Jika p-value > 0.05, data tidak stasioner; pertimbangkan differencing.")


if __name__ == "__main__":
    main()

