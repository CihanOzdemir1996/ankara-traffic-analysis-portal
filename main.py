import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Ankara Traffic Analysis Portal",
    page_icon="🚗",
    layout="wide"
)

# --- DATA INGESTION ---
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("ankara_traffic_data.csv")
        df.columns = df.columns.str.strip()
        
        # Mapping weather to numeric for regression analysis
        weather_map = {"Güneşli": 1, "Bulutlu": 2, "Yağmurlu": 3, "Karlı": 4}
        df['weather_numeric'] = df['weather_condition'].map(weather_map).fillna(1)
        
        # Data cleaning
        df = df.dropna(subset=['density_score', 'average_speed', 'weather_numeric'])
        return df, weather_map
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), {}

df, weather_map = load_and_preprocess_data()

if not df.empty:
    # --- SIDEBAR CONTROL PANEL ---
    st.sidebar.header("📍 Control Panel")
    road_list = sorted(df["road_name"].unique())
    selected_road = st.sidebar.selectbox("Select Road:", road_list)
    filtered_data = df[df["road_name"] == selected_road]

    # --- MAIN HEADER ---
    st.title("🚗 Ankara Traffic Analysis & AI Portal")
    st.markdown("---")

    # --- KEY PERFORMANCE INDICATORS (KPIs) ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Road", selected_road)
    m2.metric("Avg Density", f"%{filtered_data['density_score'].mean():.1f}")
    m3.metric("Avg Speed", f"{filtered_data['average_speed'].mean():.1f} km/h")
    m4.metric("Data Status", "Verified ✅")

    # --- VISUAL ANALYTICS ---
    col_map, col_chart = st.columns(2)
    with col_map:
        st.subheader("📍 Spatial Distribution")
        st.map(filtered_data, size='density_score', color='#ff4b4b')
    with col_chart:
        st.subheader("📊 Hourly Density Trends")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=filtered_data, x='hour', y='density_score', marker='o', color='#ff4b4b', ax=ax)
        ax.set_ylabel("Density Score (%)")
        st.pyplot(fig)

    # --- AI PREDICTION & STATISTICAL VALIDATION ---
    st.markdown("---")
    st.header("🔮 AI Prediction & Statistical Insights")
    
    # Model Training
    X = df[['density_score', 'weather_numeric']].values
    y = df['average_speed'].values

    if len(X) > 0:
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        
        col_pred, col_stats = st.columns(2)
        with col_pred:
            st.subheader("🤖 Speed Prediction Engine")
            user_density = st.slider("Target Density (%)", 0, 100, 50)
            user_weather = st.selectbox("Weather Condition", list(weather_map.keys()))
            
            prediction = model.predict(np.array([[user_density, weather_map[user_weather]]]))[0]
            st.success(f"**Predicted Average Speed:** {prediction:.2f} km/h")
            
            # Actionable Advice
            if prediction < 40:
                st.warning("🚨 **Advice:** High congestion predicted. Expect delays.")
            else:
                st.info("✅ **Advice:** Flow is predicted to be optimal.")

        with col_stats:
            st.subheader("🧪 Model Validation")
            r2 = model.score(X, y)
            dw = durbin_watson(y - y_pred)
            
            st.metric("R² Score (Accuracy)", f"{r2:.4f}")
            st.metric("Durbin-Watson Score", f"{dw:.2f}")
            
            if 1.5 < dw < 2.5:
                st.caption("✅ **Assumption Met:** No significant autocorrelation detected.")
            else:
                st.caption("⚠️ **Caution:** Potential autocorrelation in time-series data.")

    # --- CORRELATION ANALYSIS ---
    st.markdown("---")
    st.subheader("🌡️ Statistical Correlation Matrix")
    corr_df = df[['density_score', 'average_speed', 'weather_numeric', 'hour']].corr()
    fig_h, ax_h = plt.subplots(figsize=(8, 3))
    sns.heatmap(corr_df, annot=True, cmap='RdYlGn', fmt=".2f", ax=ax_h)
    st.pyplot(fig_h)

    # --- FINAL POLISHING (FOOTER) ---
    st.markdown("---")
    footer_html = """
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #666666; font-size: 14px;'>
            <b>Ankara Traffic Analysis Portal</b> | Final Version 1.0
        </p>
        <p style='color: #888888; font-size: 13px;'>
            Developed by <a href='https://www.linkedin.com/in/ozdemircihan/' target='_blank' style='text-decoration: none; color: #ff4b4b;'>Cihan Özdemir</a> 
            • <a href='https://github.com/CihanOzdemir1996' target='_blank' style='text-decoration: none; color: #ff4b4b;'>GitHub Repository</a>
        </p>
        <p style='color: #aaaaaa; font-size: 11px;'>This dashboard is for demonstration purposes using synthetic traffic data.</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

else:
    st.error("Data source could not be initialized. Please check the CSV file.")
