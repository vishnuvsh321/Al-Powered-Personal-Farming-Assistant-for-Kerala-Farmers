import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Farming Assistant", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: green;'>🌱 AI-Powered Personal Farming Assistant for Kerala Farmers</h1>
""", unsafe_allow_html=True)

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    # Normalize column names by stripping whitespace
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# -------------------------------
# Sidebar Navigation
# -------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🤖 Crop Yield Predictor", "🌾 Crop Recommendation"]
)

# -------------------------------
# 1️⃣ Dashboard – Analytics
# -------------------------------
if menu == "📊 Dashboard":
    st.subheader("📊 Agriculture Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Production (tonnes)", f"{df['Production'].sum():,.0f}")

    with col2:
        st.metric("Total Area (ha)", f"{df['Area'].sum():,.0f}")

    with col3:
        st.metric("Unique Crops", df["Crop"].nunique())

    st.markdown("### 📍 Production by Crop")
    fig_crop = px.bar(
        df.groupby("Crop")["Production"].sum().sort_values(ascending=False),
        labels={"value": "Production (tonnes)", "index": "Crop"},
        title="Crop-wise Production"
    )
    st.plotly_chart(fig_crop, width='stretch')

    st.markdown("### 🗺️ Production by State")
    fig_state = px.bar(
        df.groupby("State")["Production"].sum(),
        title="State-wise Production",
        labels={"value": "Production (tonnes)", "index": "State"}
    )
    st.plotly_chart(fig_state, width='stretch')

    st.markdown("### 📈 Production Trend by Year")
    fig_year = px.line(
        df.groupby("Crop_Year")["Production"].sum().reset_index(),
        x="Crop_Year", y="Production",
        title="Production Trend Over Years"
    )
    st.plotly_chart(fig_year, width='stretch')

# -------------------------------
# 2️⃣ Crop Yield Predictor (AI Model)
# -------------------------------
if menu == "🤖 Crop Yield Predictor":
    st.subheader("🤖 AI Model: Crop Yield Prediction")

    # Encode categorical variables
    df_model = df.copy()
    le_state = LabelEncoder()
    le_season = LabelEncoder()
    le_crop = LabelEncoder()

    df_model["State_enc"] = le_state.fit_transform(df["State"])
    df_model["Season_enc"] = le_season.fit_transform(df["Season"])
    df_model["Crop_enc"] = le_crop.fit_transform(df["Crop"])

    X = df_model[["Crop_Year", "Area", "State_enc", "Season_enc", "Crop_enc"]]
    y = df_model["Production"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    # User Inputs
    st.markdown("### 🔧 Enter Inputs")
    year = st.number_input("Year", min_value=1980, max_value=2050, value=2025)
    area = st.number_input("Area (ha)", min_value=1.0, max_value=50000.0, value=500.0)
    state = st.selectbox("State", df["State"].unique())
    season = st.selectbox("Season", df["Season"].unique())
    crop = st.selectbox("Crop", df["Crop"].unique())

    if st.button("Predict Yield"):
        input_data = pd.DataFrame({
            "Crop_Year": [year],
            "Area": [area],
            "State_enc": [le_state.transform([state])[0]],
            "Season_enc": [le_season.transform([season])[0]],
            "Crop_enc": [le_crop.transform([crop])[0]]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"🌾 **Predicted Production:** {prediction:,.2f} tonnes")

        test_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        st.info(f"Model RMSE: {rmse:,.2f}")

# -------------------------------
# 3️⃣ Crop Recommendation System
# -------------------------------
if menu == "🌾 Crop Recommendation":
    st.subheader("🌾 AI-Powered Crop Recommendation")

    state_choice = st.selectbox("Select State", df["State"].unique())
    season_choice = st.selectbox("Select Season", df["Season"].unique())
    area_choice = st.number_input("Enter Area (ha)", min_value=1.0, value=100.0)

    # Simple rule-based recommender using avg productivity
    df_filtered = df[(df["State"] == state_choice) & (df["Season"] == season_choice)]

    if not df_filtered.empty:
        df_filtered["Productivity"] = df_filtered["Production"] / df_filtered["Area"]

        top_crop = df_filtered.groupby("Crop")["Productivity"].mean().sort_values(ascending=False).head(1)

        recommended_crop = top_crop.index[0]
        productivity = top_crop.values[0]

        st.success(f"🌟 **Recommended Crop:** {recommended_crop}")
        st.write(f"Expected Productivity: **{productivity:.2f} tonnes/ha**")
    else:
        st.warning("No data available for selected conditions.")
