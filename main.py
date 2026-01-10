import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Ankara AI & Data Science Portal", layout="wide", page_icon="🏎️")

# --- 2. DATA SCIENCE AUDIT (BACKEND) ---
@st.cache_data
def load_and_audit_data():
    try:
        df = pd.read_csv("ankara_traffic_data.csv")
        df.columns = df.columns.str.strip()
        weather_map = {"Güneşli": 1, "Bulutlu": 2, "Yağmurlu": 3, "Karlı": 4}
        df['weather_numeric'] = df['weather_condition'].map(weather_map).fillna(1)
        
        # Outlier Removal using IQR Method
        Q1, Q3 = df['average_speed'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_cleaned = df[(df['average_speed'] >= Q1 - 1.5 * IQR) & (df['average_speed'] <= Q3 + 1.5 * IQR)].copy()
        return df, df_cleaned, weather_map
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

df_raw, df, weather_map = load_and_audit_data()

if not df.empty:
    # --- SIDEBAR CONTROL PANEL ---
    st.sidebar.title("🎮 Filter & Control")
    road_list = sorted(df["road_name"].unique())
    selected_road = st.sidebar.selectbox("Select Arterial Road:", road_list)
    road_data = df[df["road_name"] == selected_road]

    # --- MAIN HEADER ---
    st.title("🚗 Ankara Traffic: AI Prediction & Data Science Portal")
    st.markdown("An end-to-end data science application featuring statistical auditing and AI-driven traffic forecasting.")
    st.divider()

    # --- SECTION 1: KEY PERFORMANCE INDICATORS (KPIs) ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Road", selected_road)
    m2.metric("Avg Density", f"%{road_data['density_score'].mean():.1f}")
    m3.metric("Avg Speed", f"{road_data['average_speed'].mean():.1f} km/h")
    m4.metric("Data Status", "Verified ✅")

    # --- SECTION 2: STATISTICAL AUDIT REPORT ---
    st.header("🔬 1. Statistical Audit & Data Cleaning")
    c1, c2, c3 = st.columns([1, 1, 1])
    
    with c1:
        st.write("### Outlier Detection")
        st.metric("Raw Observations", len(df_raw))
        st.metric("Cleaned Data", len(df))
        st.caption(f"{(len(df_raw)-len(df))} outliers removed via IQR method.")

    with c2:
        st.write("### Normality Test")
        stat, p_norm = stats.shapiro(df['average_speed'][:500])
        st.metric("Shapiro-Wilk p-value", f"{p_norm:.4f}")
        is_normal = p_norm > 0.05
        if is_normal:
            st.success("✅ Normal Distribution detected. Pearson Parametric tests will be used.")
            corr_method = 'pearson'
        else:
            st.warning("⚠️ Non-Normal Distribution detected. Spearman Non-Parametric tests will be used.")
            corr_method = 'spearman'

    with c3:
        st.write("### Speed Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        sns.histplot(df['average_speed'], kde=True, color="#ff4b4b", ax=ax_hist)
        ax_hist.set_xlabel("Average Speed (km/h)")
        st.pyplot(fig_hist)

    # --- SECTION 3: STATISTICAL RELATIONSHIPS ---
    st.divider()
    st.header("🌡️ 2. Statistical Relationships & Heatmap")
    
    col_heat, col_sig = st.columns([1.2, 0.8])
    with col_heat:
        fig_h, ax_h = plt.subplots(figsize=(10, 5))
        sns.heatmap(df[['density_score', 'average_speed', 'weather_numeric', 'hour']].corr(method=corr_method), 
                    annot=True, cmap='coolwarm', ax=ax_h)
        st.pyplot(fig_h)
    
    with col_sig:
        st.write("### Significance Analysis")
        if corr_method == 'pearson':
            r_val, p_val = stats.pearsonr(df['density_score'], df['average_speed'])
        else:
            r_val, p_val = stats.spearmanr(df['density_score'], df['average_speed'])
        
        st.write(f"**Method:** {corr_method.capitalize()} Correlation")
        st.write(f"**Coefficient (R):** {r_val:.4f}")
        st.write(f"**P-Value:** {p_val:.2e}")
        if p_val < 0.05:
            st.success("✅ Statistically Significant relationship confirmed.")
        else:
            st.error("❌ Relationship is not statistically significant.")

    # --- SECTION 4: AI PREDICTION ENGINE ---
    st.divider()
    st.header("🤖 3. AI Prediction Engine")
    
    X = df[['density_score', 'weather_numeric']].values
    y = df['average_speed'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    ca1, ca2 = st.columns(2)
    with ca1:
        st.subheader("Traffic Simulator")
        u_dens = st.slider("Select Density (%)", 0, 100, 50)
        u_weath = st.selectbox("Select Weather Condition", list(weather_map.keys()))
        pred = model.predict(np.array([[u_dens, weather_map[u_weath]]]))[0]
        st.success(f"**AI
