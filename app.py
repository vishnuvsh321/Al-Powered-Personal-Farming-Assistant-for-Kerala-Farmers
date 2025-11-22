import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# ===========================
# Streamlit Page Config
# ===========================
st.set_page_config(
    page_title="AI Crop Yield Analytics Dashboard",
    layout="wide",
    page_icon="🌾"
)

# ===========================
# Custom Theme (White + Green)
# ===========================
st.markdown("""
<style>
body {
    background-color: #ffffff;
}
.block-container {
    padding-top: 20px;
}
h1, h2, h3 {
    color: #2e7d32 !important;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# Load Data
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_data()

# ===========================
# Title
# ===========================
st.title("🌾 AI-Powered Crop Yield & Price Analytics Dashboard")

st.write("This dashboard provides analytics, visualizations, and AI forecasting models using uploaded data.")

# ===========================
# Sidebar Controls
# ===========================
st.sidebar.header("🔧 Controls")

show_stats = st.sidebar.checkbox("Show Summary Statistics", True)
show_moving_avg = st.sidebar.checkbox("Show Moving Average", False)
ma_window = st.sidebar.slider("MA Window", 3, 60, 12)

show_trend = st.sidebar.checkbox("Show Trend Line", False)

forecast_months = st.sidebar.number_input("Forecast Months", 1, 36, 6)
model_choice = st.sidebar.selectbox(
    "Choose Forecast Model",
    ["ARIMA", "Prophet", "Linear Regression"]
)

# ===========================
# Main Chart
# ===========================
st.subheader("📈 Crop Yield Trend")

fig = px.line(df, x="Date", y="Yield", title="Crop Yield Over Time",
              template="plotly_white", color_discrete_sequence=["green"])
st.plotly_chart(fig, use_container_width=True)

# ===========================
# Moving Average
# ===========================
if show_moving_avg:
    st.subheader("📉 Moving Average Smoothing")
    df["MA"] = df["Yield"].rolling(ma_window).mean()
    fig_ma = px.line(df, x="Date", y=["Yield", "MA"],
                     labels={"value": "Yield"},
                     title=f"{ma_window}-Point Moving Average",
                     template="plotly_white")
    st.plotly_chart(fig_ma, use_container_width=True)

# ===========================
# Summary Statistics
# ===========================
if show_stats:
    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

# ===========================
# Trend Line (Linear Regression)
# ===========================
if show_trend:
    st.subheader("📈 Trend Line (Regression)")
    df["t"] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[["t"]], df["Yield"])
    df["Trend"] = model.predict(df[["t"]])

    fig_trend = px.line(df, x="Date", y=["Yield", "Trend"],
                        title="Yield Trend Line",
                        template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True)

# ===========================
# Forecasting Models
# ===========================
st.subheader("🔮 AI Forecasting")

def forecast_arima(df, periods):
    model = ARIMA(df["Yield"], order=(5,1,0))
    model_fit = model.fit()
    return model_fit.forecast(periods)

def forecast_prophet(df, periods):
    pf = df[["Date", "Yield"]].rename(columns={"Date": "ds", "Yield": "y"})
    m = Prophet()
    m.fit(pf)
    future = m.make_future_dataframe(periods=periods, freq="M")
    forecast = m.predict(future)
    forecast.index = forecast["ds"]
    return forecast["yhat"]

def forecast_linear(df, periods):
    df["t"] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[["t"]], df["Yield"])
    future_t = np.arange(len(df), len(df) + periods)
    preds = model.predict(future_t.reshape(-1, 1))
    return pd.Series(preds)

# Run the selected model
if model_choice == "ARIMA":
    forecast = forecast_arima(df, forecast_months)
elif model_choice == "Prophet":
    forecast = forecast_prophet(df, forecast_months)
else:
    forecast = forecast_linear(df, forecast_months)

# Display Forecast
st.write(forecast)

# Plot Forecast
forecast_df = pd.DataFrame({
    "Date": pd.date_range(df["Date"].iloc[-1], periods=forecast_months+1, freq="M")[1:],
    "Forecast": forecast.values
})

fig_fc = px.line(forecast_df, x="Date", y="Forecast",
                 title=f"{model_choice} Forecast for Next {forecast_months} Months",
                 template="plotly_white",
                 color_discrete_sequence=["green"])
st.plotly_chart(fig_fc, use_container_width=True)

# ===========================
# Footer
# ===========================
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Prophet, ARIMA & ML.")

