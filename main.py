import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Ankara Traffic: Data Science Audit", layout="wide", page_icon="🔬")

# --- 2. DATA INGESTION & AUDIT ---
@st.cache_data
def load_and_audit_data():
    try:
        df = pd.read_csv("ankara_traffic_data.csv")
        df.columns = df.columns.str.strip()
        weather_map = {"Güneşli": 1, "Bulutlu": 2, "Yağmurlu": 3, "Karlı": 4}
        df['weather_numeric'] = df['weather_condition'].map(weather_map).fillna(1)
        
        # [Data Scientist Step] Outlier Removal (IQR)
        Q1, Q3 = df['average_speed'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_cleaned = df[(df['average_speed'] >= Q1 - 1.5 * IQR) & (df['average_speed'] <= Q3 + 1.5 * IQR)].copy()
        return df, df_cleaned, weather_map
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}

df_raw, df, weather_map = load_and_audit_data()

if not df.empty:
    # --- HEADER ---
    st.title("🔬 Ankara Traffic: Data Science Audit & AI Prediction")
    st.markdown("Bu portal, verinin istatistiksel geçerliliğini denetler ve AI modellerini bilimsel standartlara göre sunar.")
    st.divider()

    # --- SECTION 1: STATISTICAL AUDIT (ÖNCELİKLİ BÖLÜM) ---
    st.header("1. Statistical Audit & Data Cleaning")
    
    # Metrikler
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Data", len(df_raw))
    c2.metric("Outliers Removed", len(df_raw) - len(df))
    
    stat, p_norm = stats.shapiro(df['average_speed'][:500])
    c3.metric("Shapiro-Wilk (p)", f"{p_norm:.4f}")

    # Dağılım ve Normallik Kararı
    col_norm1, col_norm2 = st.columns(2)
    with col_norm1:
        fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
        sns.histplot(df['average_speed'], kde=True, color="#2E86C1", ax=ax_dist)
        ax_dist.set_title("Speed Distribution After Cleaning")
        st.pyplot(fig_dist)
    
    with col_norm2:
        st.subheader("Decision Logic")
        is_normal = p_norm > 0.05
        if is_normal:
            st.success("✅ Veri Normal Dağılıyor. Pearson Parametrik Testleri kullanılacaktır.")
            corr_method = 'pearson'
        else:
            st.warning("⚠️ Veri Normal Dağılımdan Sapma Gösteriyor. Spearman Non-Parametrik Testleri kullanılacaktır.")
            corr_method = 'spearman'
        
        # P-Value & Correlation
        if corr_method == 'pearson':
            r_val, p_val = stats.pearsonr(df['density_score'], df['average_speed'])
        else:
            r_val, p_val = stats.spearmanr(df['density_score'], df['average_speed'])
            
        st.write(f"**Correlation ({corr_method.capitalize()}):** {r_val:.4f}")
        st.write(f"**Statistical Significance (p):** {p_val:.2e}")

    # --- SECTION 2: AI MODELLING & DIAGNOSTICS ---
    st.divider()
    st.header("2. AI Prediction & Model Diagnostics")
    
    X = df[['density_score', 'weather_numeric']].values
    y = df['average_speed'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    cp1, cp2 = st.columns(2)
    with cp1:
        st.write("### 🤖 Prediction Engine")
        u_dens = st.slider("Select Density (%)", 0, 100, 50)
        u_weath = st.selectbox("Select Weather", list(weather_map.keys()))
        pred = model.predict(np.array([[u_dens, weather_map[u_weath]]]))[0]
        st.success(f"**Predicted Speed:** {pred:.2f} km/h")

    with cp2:
        st.write("### 🧪 Integrity Checks")
        dw = durbin_watson(y - y_pred)
        st.metric("R² (Confidence)", f"{model.score(X, y):.4f}")
        st.metric("Durbin-Watson", f"{dw:.2f}")
        if 1.5 < dw < 2.5:
            st.caption("✅ Hatalar arasında otokorelasyon yok.")
        else:
            st.caption("⚠️ Otokorelasyon riski: Zaman serisi bağımlılığı olabilir.")

    # --- SECTION 3: VISUALIZATIONS & SPATIAL DATA (GÖRSEL BÖLÜM) ---
    st.divider()
    st.header("3. Geospatial & Visual Analytics")
    
    # Korelasyon Isı Haritası
    st.subheader("Statistical Heatmap")
    fig_h, ax_h = plt.subplots(figsize=(10, 4))
    sns.heatmap(df[['density_score', 'average_speed', 'weather_numeric', 'hour']].corr(method=corr_method), 
                annot=True, cmap='coolwarm', ax=ax_h)
    st.pyplot(fig_h)
    

    # Harita ve Saatlik Grafik
    st.subheader(f"Route Specific Analysis")
    selected_road = st.sidebar.selectbox("Filter by Road (Visuals):", sorted(df["road_name"].unique()))
    road_data = df[df["road_name"] == selected_road]

    col_map, col_trend = st.columns([1.2, 0.8])
    with col_map:
        st.map(road_data, size='density_score', color='#ff4b4b')
        
    with col_trend:
        fig_trend, ax_trend = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=road_data, x='hour', y='density_score', marker='o', color='#ff4b4b', ax=ax_trend)
        ax_trend.set_title(f"{selected_road} Hourly Density")
        st.pyplot(fig_trend)
        

    # --- FOOTER ---
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center;'>
        <b>Ankara Traffic AI Portal</b> | Built by <a href='https://www.linkedin.com/in/ozdemircihan/' target='_blank'>Cihan Özdemir</a>
        <br><small>Verified with Interquartile Range (IQR) and Shapiro-Wilk Tests</small>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Data load failed.")
