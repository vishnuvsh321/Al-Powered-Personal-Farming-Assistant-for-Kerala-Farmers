import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# -----------------------------------------------------------
# PAGE CONFIG + CSS STYLING
# -----------------------------------------------------------
st.set_page_config(page_title="AI For Indian Farmers", layout="wide")

st.markdown("""
<style>
/* MAIN BACKGROUND */
body {
    background-color: #ffffff;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #e8f5e9;
}

/* HEADER */
h1, h2, h3 {
    color: #1b5e20 !important;
}

/* METRICS */
.metric {
    background-color: #f1f8e9 !important;
    padding: 15px;
    border-radius: 10px;
}

/* SELECTBOX LABEL */
.css-1pahdxg-control {
    border: 2px solid #2e7d32 !important;
}

/* BUTTON GREEN THEME */
.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #1b5e20;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("🌿 Navigation")
page = st.sidebar.selectbox(
    "Go to",
    ["🏠 Home", "📊 Dashboard", "🤖 Yield Prediction", "🌾 Crop Recommendation"]
)

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if page == "🏠 Home":
    st.markdown("<h1>🌱 AI Farming Assistant for Indian Farmers</h1>", unsafe_allow_html=True)

    st.image(
        "https://wallpapercave.com/wp/wp5627799.jpg",
        use_column_width=True,
        caption="Agriculture • India • Sustainability"
    )

    st.markdown("""
    ### 🇮🇳 Empowering Indian Farmers with AI  
    This platform provides:
    - 📊 **Interactive Agriculture Analytics Dashboard**  
    - 🤖 **AI-Powered Yield Prediction**  
    - 🌾 **Smart Crop Recommendation System**  
    - 🧠 Data insights to support scientific decisions  
    """)


# ---------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------
elif section == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")

    # ------------------------------
    # Filter Slicers
    # ------------------------------
    st.subheader("🔎 Filters (Slicers)")

    # Numeric slicers
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    col1, col2, col3 = st.columns(3)

    with col1:
        num_filter_col = st.selectbox("Select Numeric Column", numeric_cols)
        num_min, num_max = st.slider(
            f"Range for {num_filter_col}",
            float(df[num_filter_col].min()),
            float(df[num_filter_col].max()),
            (float(df[num_filter_col].min()), float(df[num_filter_col].max()))
        )

    with col2:
        cat_filter_col = st.selectbox("Select Categorical Column", df.select_dtypes(include=['object']).columns)
        cat_unique = df[cat_filter_col].unique().tolist()
        cat_selection = st.multiselect("Select Categories", cat_unique, default=cat_unique)

    with col3:
        if date_cols:
            date_col_selected = st.selectbox("Select Date Column", date_cols)
            date_min, date_max = st.date_input(
                f"Select Date Range for {date_col_selected}",
                [df[date_col_selected].min(), df[date_col_selected].max()]
            )
        else:
            date_col_selected = None

    # APPLY FILTERS
    filtered_df = df[
        (df[num_filter_col] >= num_min) &
        (df[num_filter_col] <= num_max) &
        (df[cat_filter_col].isin(cat_selection))
    ]

    if date_col_selected:
        filtered_df = filtered_df[
            (filtered_df[date_col_selected] >= pd.to_datetime(date_min)) &
            (filtered_df[date_col_selected] <= pd.to_datetime(date_max))
        ]

    st.write("### Filtered Data")
    st.dataframe(filtered_df, use_container_width=True)

    # ------------------------------
    # Dashboard Plots
    # ------------------------------
    st.subheader("📈 Visual Insights")

    colA, colB = st.columns(2)

    with colA:
        numeric_choice = st.selectbox("Select Column for Histogram", numeric_cols)
        fig1 = px.histogram(filtered_df, x=numeric_choice, nbins=30, title=f"Distribution of {numeric_choice}")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        if len(numeric_cols) >= 2:
            num_x = st.selectbox("X-axis", numeric_cols, index=0)
            num_y = st.selectbox("Y-axis", numeric_cols, index=1)
            fig2 = px.scatter(filtered_df, x=num_x, y=num_y, trendline="ols",
                              title=f"{num_x} vs {num_y}")
            st.plotly_chart(fig2, use_container_width=True)


# -----------------------------------------------------------
# YIELD PREDICTION MODEL
# -----------------------------------------------------------
if page == "🤖 Yield Prediction":
    st.header("🤖 AI Model: Crop Yield Prediction")

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

    model = RandomForestRegressor(n_estimators=150)
    model.fit(X_train, y_train)

    st.subheader("Enter Inputs for Prediction")

    year = st.number_input("Crop Year", min_value=1990, max_value=2050, value=2024)
    area = st.number_input("Area (ha)", min_value=1.0, value=500.0)
    state = st.selectbox("State", df["State"].unique())
    season = st.selectbox("Season", df["Season"].unique())
    crop = st.selectbox("Crop", df["Crop"].unique())

    if st.button("Predict Yield"):
        user_input = pd.DataFrame({
            "Crop_Year": [year],
            "Area": [area],
            "State_enc": [le_state.transform([state])[0]],
            "Season_enc": [le_season.transform([season])[0]],
            "Crop_enc": [le_crop.transform([crop])[0]],
        })

        pred = model.predict(user_input)[0]
        st.success(f"🌾 **Predicted Yield: {pred:,.2f} tonnes**")

        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        st.info(f"Model RMSE: {rmse:,.2f}")

# -----------------------------------------------------------
# CROP RECOMMENDATION SYSTEM
# -----------------------------------------------------------
if page == "🌾 Crop Recommendation":
    st.header("🌾 Smart Crop Recommendation")

    state_sel = st.selectbox("Select State", df["State"].unique())
    season_sel = st.selectbox("Select Season", df["Season"].unique())
    area_sel = st.number_input("Available Area (ha)", min_value=1.0, value=100.0)

    df_f = df[(df["State"] == state_sel) & (df["Season"] == season_sel)]

    if df_f.empty:
        st.warning("⚠ No data available for selected filters.")
    else:
        df_f["Productivity"] = df_f["Production"] / df_f["Area"]

        top = df_f.groupby("Crop")["Productivity"].mean().sort_values(ascending=False).head(1)

        recommended_crop = top.index[0]
        prod_val = top.values[0]

        st.success(f"🌟 **Recommended Crop: {recommended_crop}**")
        st.info(f"Expected Productivity: **{prod_val:.2f} tonnes/ha**")

