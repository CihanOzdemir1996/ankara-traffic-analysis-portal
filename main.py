import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Ankara Traffic: Ultimate AI Portal", layout="wide", page_icon="🔬")

# --- 2. ADVANCED DATA AUDIT (BACKEND) ---
@st.cache_data
def load_and_audit_data():
    try:
        df = pd.read_csv("ankara_traffic_data.csv")
        df.columns = df.columns.str.strip()
        weather_map = {"Güneşli": 1, "Bulutlu": 2, "Yağmurlu": 3, "Karlı": 4}
        df['weather_numeric'] = df['weather_condition'].map(weather_map).fillna(1)
        
        # [DS Step] Outlier Removal (IQR)
        Q1, Q3 = df['average_speed'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_cleaned = df[(df['average_speed'] >= Q1 - 1.5 * IQR) & (df['average_speed'] <= Q3 + 1.5 * IQR)].copy()
        return df, df_cleaned, weather_map
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

df_raw, df, weather_map = load_and_audit_data()

if not df.empty:
    # --- SIDEBAR CONTROL ---
    st.sidebar.title("🎮 Analysis Controls")
    selected_road = st.sidebar.selectbox("Focus Road (Visuals):", sorted(df["road_name"].unique()))
    road_data = df[df["road_name"] == selected_road]

    # --- HEADER ---
    st.title("🚗 Ankara Traffic: AI Prediction & Deep Statistical Audit")
    st.markdown("This platform integrates **Software Engineering, AI Modeling, and Advanced Econometric Testing.**")
    st.divider()

    # --- SECTION 1: DATA INTEGRITY & NORMALITY ---
    st.header("🔬 1. Data Integrity & Normality Audit")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.write("### Cleaning (IQR)")
        st.metric("Raw Observations", len(df_raw))
        st.metric("Cleaned Data", len(df))
        st.caption(f"{len(df_raw)-len(df)} outliers removed to reduce noise.")

    with c2:
        st.write("### Normality (Shapiro)")
        stat_sw, p_norm = stats.shapiro(df['average_speed'][:500])
        st.metric("Shapiro-Wilk p-value", f"{p_norm:.4f}")
        is_normal = p_norm > 0.05
        corr_method = 'pearson' if is_normal else 'spearman'
        st.info(f"Method: **{corr_method.capitalize()}**")

    with c3:
        st.write("### Speed Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        sns.histplot(df['average_speed'], kde=True, color="#ff4b4b", ax=ax_hist)
        st.pyplot(fig_hist)

    # --- SECTION 2: MULTICOLLINEARITY & VARIANCE (THE ADVANCED TESTS) ---
    st.divider()
    st.header("🧪 2. Multicollinearity (VIF) & Homoscedasticity")
    
    # Model for testing
    X_audit = df[['density_score', 'weather_numeric', 'hour']]
    X_audit_const = sm.add_constant(X_audit)
    y_audit = df['average_speed']
    model_audit = sm.OLS(y_audit, X_audit_const).fit()

    cv1, cv2 = st.columns(2)
    with cv1:
        st.write("### Multicollinearity (VIF)")
        vif_df = pd.DataFrame()
        vif_df["Feature"] = X_audit.columns
        vif_df["VIF"] = [variance_inflation_factor(X_audit.values, i) for i in range(len(X_audit.columns))]
        st.table(vif_df)
        if vif_df["VIF"].max() < 5:
            st.success("✅ No Multicollinearity detected (VIF < 5).")

    with cv2:
        st.write("### Homoscedasticity (Breusch-Pagan)")
        bp_test = het_breuschpagan(model_audit.resid, model_audit.model.exog)
        p_bp = bp_test[1]
        st.metric("Breusch-Pagan p-value", f"{p_bp:.4f}")
        if p_bp > 0.05:
            st.success("✅ Variance is Homogeneous.")
        else:
            st.warning("⚠️ Heteroscedasticity detected (Non-constant variance).")

    # --- SECTION 3: AI PREDICTION ENGINE ---
    st.divider()
    st.header("🤖 3. AI Prediction & Model Integrity")
    
    model_ai = LinearRegression().fit(X_audit, y_audit)
    
    cp1, cp2 = st.columns(2)
    with cp1:
        st.subheader("Simulator")
        u_dens = st.slider("Density (%)", 0, 100, 50)
        u_weath = st.selectbox("Weather", list(weather_map.keys()))
        u_hour = st.slider("Hour", 0, 23, 12)
        pred = model_ai.predict([[u_dens, weather_map[u_weath], u_hour]])[0]
        st.success(f"**AI Predicted Speed:** {pred:.2f} km/h")

    with cp2:
        st.subheader("Diagnostics")
        dw = durbin_watson(model_audit.resid)
        st.metric("R² Confidence", f"{model_ai.score(X_audit, y_audit):.4f}")
        st.metric("Durbin-Watson", f"{dw:.2f}")

    # --- SECTION 4: VISUAL ANALYTICS (SOFTWARE & UI) ---
    st.divider()
    st.header("📊 4. Visual & Spatial Analytics")
    
    # Heatmap
    st.subheader("Statistical Relationship Matrix")
    fig_h, ax_h = plt.subplots(figsize=(10, 4))
    sns.heatmap(df[['density_score', 'average_speed', 'weather_numeric', 'hour']].corr(method=corr_method), 
                annot=True, cmap='RdYlGn', ax=ax_h)
    st.pyplot(fig_h)

    # Map & Trends
    col_map, col_line = st.columns([1.2, 0.8])
    with col_map:
        st.subheader(f"📍 Density Map: {selected_road}")
        st.map(road_data, size='density_score', color='#ff4b4b')
        
    with col_line:
        st.subheader("📈 Hourly Trend")
        fig_trend, ax_trend = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=road_data, x='hour', y='density_score', marker='o', color='#ff4b4b', ax=ax_trend)
        st.pyplot(fig_trend)
        

    # --- FOOTER ---
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #888888;'>
        <b>Ankara Traffic AI Portal</b> | Developed by <a href='https://www.linkedin.com/in/ozdemircihan/' target='_blank'>Cihan Özdemir</a>
        <br><small>Advanced Audit: IQR, Shapiro-Wilk, VIF, Breusch-Pagan, Durbin-Watson</small>
    </div>
    """, unsafe_allow_html=True)
