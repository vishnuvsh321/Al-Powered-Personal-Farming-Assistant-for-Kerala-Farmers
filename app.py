import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from statsmodels.tsa.arima.model import ARIMA

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="Indian Farmers AI Dashboard", layout="wide")

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")   # your file
    # Fix date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

df = load_data()

# ============================
# CSS STYLING
# ============================
st.markdown("""
<style>

.navbar {
    background-color: #1E88E5;
    padding: 15px;
    border-radius: 10px;
    width: 100%;
}

.nav-title {
    color: white;
    font-size: 24px;
    font-weight: bold;
    padding-left: 10px;
}

.dropdown {
    position: relative;
    display: inline-block;
    float: right;
    margin-right: 20px;
}

.dropbtn {
    background-color: white;
    color: #1E88E5;
    padding: 12px 20px;
    font-size: 16px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
}

.dropbtn:hover {
    background-color: #e3f2fd;
}

.dropdown-content {
    display: none;
    position: absolute;
    background-color: white;
    min-width: 180px;
    box-shadow: 0px 8px 16px rgba(0,0,0,0.2);
    z-index: 5;
    border-radius: 8px;
}

.dropdown-content a {
    color: #1E88E5;
    padding: 12px 16px;
    text-decoration: none;
    display: block;
    font-weight: 500;
}

.dropdown-content a:hover {
    background-color: #e3f2fd;
}

.dropdown:hover .dropdown-content {
    display: block;
}

</style>
""", unsafe_allow_html=True)

# ============================
# NAVBAR HTML
# ============================
st.markdown("""
<div class="navbar">
    <span class="nav-title">Indian Farmers Dashboard</span>
    <div class="dropdown">
        <button class="dropbtn">Navigate ▼</button>
        <div class="dropdown-content">
            <a href="/?page=Home">Home</a>
            <a href="/?page=Analytics">Analytics</a>
            <a href="/?page=Crop_Recommendation">AI Crop Recommendation</a>
            <a href="/?page=Forecasting">Yield Forecasting</a>
            <a href="/?page=About">About</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

query_params = st.experimental_get_query_params()
page = query_params.get("page", ["Home"])[0]


# ============================
# HOME PAGE
# ============================
if page == "Home":
    st.title("🌾 Welcome to the Indian Farmers AI Dashboard")
    st.write("A unified platform for analytics, AI-powered crop recommendations, and yield forecasting.")


# ============================
# ANALYTICS PAGE
# ============================
elif page == "Analytics":
    st.title("📊 Data Analytics Dashboard")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Yield Trend Over Time")
    if "date" in df.columns and "yield" in df.columns:
        fig = px.line(df, x="date", y="yield", title="Yield Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Dataset missing 'date' or 'yield' column.")

    st.write("### Compare Crops")
    if "crop" in df.columns and "yield" in df.columns:
        crop_avg = df.groupby("crop")["yield"].mean().reset_index()
        fig = px.bar(crop_avg, x="crop", y="yield", title="Average Yield by Crop")
        st.plotly_chart(fig, use_container_width=True)


# ============================
# AI CROP RECOMMENDATION
# ============================
elif page == "Crop_Recommendation":
    st.title("🤖 AI-Powered Crop Recommendation System")

    required_cols = ["nitrogen", "phosphorus", "potassium", "ph", "rainfall", "crop"]

    if all(col in df.columns for col in required_cols):

        # Encode crop labels
        le = LabelEncoder()
        df["crop_label"] = le.fit_transform(df["crop"])

        # Features & labels
        X = df[["nitrogen", "phosphorus", "potassium", "ph", "rainfall"]]
        y = df["crop_label"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.success(f"Model Trained Successfully! Accuracy: {accuracy*100:.2f}%")

        st.write("### Enter Soil Parameters")

        n = st.number_input("Nitrogen", 0, 200)
        p = st.number_input("Phosphorus", 0, 200)
        k = st.number_input("Potassium", 0, 200)
        ph = st.number_input("pH Level", 0.0, 14.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0)

        if st.button("Recommend Crop"):
            input_data = np.array([[n, p, k, ph, rainfall]])
            pred = model.predict(input_data)[0]
            crop = le.inverse_transform([pred])[0]

            st.success(f"🌱 Recommended Crop: **{crop}**")

    else:
        st.error("Dataset missing necessary columns for crop recommendation model.")


# ============================
# FORECASTING PAGE
# ============================
elif page == "Forecasting":
    st.title("📈 Future Yield Forecasting (ARIMA Model)")

    if "date" in df.columns and "yield" in df.columns:
        ts = df.set_index("date")["yield"].dropna()

        try:
            model = ARIMA(ts, order=(1,1,1))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=12)
            forecast_df = pd.DataFrame({
                "Date": pd.date_range(start=df["date"].max(), periods=12, freq="M"),
                "Forecast_Yield": forecast
            })

            fig = px.line(forecast_df, x="Date", y="Forecast_Yield", title="Yield Forecast for Next 12 Months")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Forecasting Error: {e}")

    else:
        st.error("Dataset missing 'date' or 'yield' column.")


# ============================
# ABOUT PAGE
# ============================
elif page == "About":
    st.title("ℹ️ About This Project")
    st.write("""
    This AI-powered dashboard helps Indian farmers through:
    - Smart data analytics  
    - Machine learning crop recommendations  
    - Time-series based yield forecasting  
    - Clean and modern UI  
    """)

