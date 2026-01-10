import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Ankara AI & Data Science Portal",
    page_icon="🏎️",
    layout="wide"
)

# --- 2. DATA SCIENCE AUDIT (BACKEND) ---
@st.cache_data
def load_and_audit_data():
    try:
        df = pd.read_csv("ankara_traffic_data.csv")
        df.columns = df.columns.str.strip()
        
        # Mapping weather to numeric for statistical modeling
        weather_map = {"Güneşli": 1, "Bulutlu": 2, "Yağmurlu": 3, "Karlı": 4}
        df['weather_numeric'] = df['weather_condition'].map(weather_map).fillna(1)
        
        # [Data Science Step] Outlier Removal using IQR Method
        Q1, Q3 = df['average_speed'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_cleaned = df[(df['average_speed'] >= Q1 - 1.5 * IQR) & (df['average_speed'] <= Q3 + 1.5 * IQR)].copy()
        
        return df, df_cleaned, weather_map
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

df_raw, df, weather_map = load_and_audit_data()

if not df.empty:
    # --- SIDEBAR CONTROL PANEL ---
    st.sidebar.title("🎮 Model Control Center")
    road_list = sorted(df["road_name"].unique())
    selected_road = st.sidebar.selectbox("Filter Visuals by Road:", road_list)
    road_data = df[df["road_name"] == selected_road]

    # --- MAIN HEADER ---
    st.title("🚗 Ankara Traffic: AI Prediction & Data Science Portal")
    st.markdown("Connecting **Statistical Rigor** with **Modern AI Software Engineering**.")
    st.divider()

    # --- SECTION 1: STATISTICAL AUDIT (The Scientist's Perspective) ---
    st.header("🔬 1. Statistical Audit & Data Integrity")
    
    # Audit Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw Observations", len(df_raw))
    m2.metric("Outliers Removed", len(df_raw) - len(df))
    
    # Shapiro-Wilk Normality Test
    stat, p_norm = stats.shapiro(df['average_speed'][:500])
    m3.metric("Shapiro-Wilk p-value", f"{p_norm:.4f}")
    
    is_normal = p_norm > 0.05
    m4.metric("Verified Status", "Normal Distribution" if is_normal else "Skewed Distribution")

    # Distribution & Logic
    col_dist, col_logic = st.columns([1.2, 0.8])
    with col_dist:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        sns.histplot(df['average_speed'], kde=True, color="#ff4b4b", ax=ax_hist)
        ax_hist.set_title("Post-Cleaning Speed Distribution")
        st.pyplot(fig_hist)

    with col_logic:
        st.subheader("Decision Intelligence")
        if is_normal:
            st.success("✅ **Parametric Assumption Met:** Using Pearson Correlation.")
            corr_method = 'pearson'
        else:
            st.warning("⚠️ **Non-Parametric Shift:** Using Spearman Rank Correlation.")
            corr_method = 'spearman'
        
        # Significance Testing
        if corr_method == 'pearson':
            r_val, p_val = stats.pearsonr(df['density_score'], df['average_speed'])
        else:
            r_val, p_val = stats.spearmanr(df['density_score'], df['average_speed'])
            
        st.write(f"**Calculated Coefficient:** {r_val:.4f}")
        st.write(f"**P-Value (Significance):** {p_val:.2e}")
        if p_val < 0.05:
            st.info("The relationship between variables is statistically significant.")

    # --- SECTION 2: AI PREDICTION ENGINE (The Engineer's Perspective) ---
    st.divider()
    st.header("🤖 2. AI Predictive Modeling & Diagnostics")
    
    X = df[['density_score', 'weather_numeric']].values
    y = df['average_speed'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    col_sim, col_diag = st.columns(2)
    with col_sim:
        st.write("### Traffic Simulator")
        u_dens = st.slider("Target Density (%)", 0, 100, 50)
        u_weath = st.selectbox("Weather Condition", list(weather_map.keys()))
        
        # FIXED LINE: f-string now properly terminated
        prediction = model.predict(np.array([[u_dens, weather_map[u_weath]]]))[0]
        st.success(f"**AI Predicted Average Speed:** {prediction:.2f} km/h")

    with col_diag:
        st.write("### Model Integrity Logs")
        dw_score = durbin_watson(y - y_pred)
        st.metric("R² Confidence Score", f"{model.score(X, y):.4f}")
        st.metric("Durbin-Watson Diagnostic", f"{dw_score:.2f}")
        if 1.5 < dw_score < 2.5:
            st.caption("✅ No significant autocorrelation detected in residuals.")

    # --- SECTION 3: VISUAL ANALYTICS (The Full-Stack Experience) ---
    st.divider()
    st.header("📊 3. Visual & Geospatial Analytics")
    
    # Heatmap
    st.subheader("Global Correlation Matrix")
    fig_heat, ax_heat = plt.subplots(figsize=(12, 4))
    sns.heatmap(df[['density_score', 'average_speed', 'weather_numeric', 'hour']].corr(method=corr_method), 
                annot=True, cmap='RdYlGn', ax=ax_heat)
    st.pyplot(fig_heat)

    # Map and Hourly Trends
    st.subheader(f"Arterial Focus: {selected_road}")
    col_map, col_trend = st.columns([1.2, 0.8])
    
    with col_map:
        st.map(road_data, size='density_score', color='#ff4b4b')
        
    with col_trend:
        fig_trend, ax_trend = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=road_data, x='hour', y='density_score', marker='o', color='#ff4b4b', ax=ax_trend)
        ax_trend.set_ylabel("Density (%)")
        st.pyplot(fig_trend)

    # --- FOOTER ---
    st.markdown("---")
    footer_html = f"""
    <div style='text-align: center; padding: 10px;'>
        <p style='color: #888888; font-size: 14px;'>
            <b>Ankara Traffic AI Portal</b> | Developed by <a href='https://www.linkedin.com/in/ozdemircihan/' target='_blank' style='color: #ff4b4b; text-decoration: none;'>Cihan Özdemir</a>
        </p>
        <p style='color: #aaaaaa; font-size: 11px;'>Validated via IQR, Shapiro-Wilk, and Durbin-Watson Statistical Frameworks.</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

else:
    st.error("Data source could not be initialized. Please check the CSV file.")
