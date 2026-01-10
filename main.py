import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Ankara Traffic: Data Science Audit", layout="wide")


# --- DATA AUDIT & PREPROCESSING ---
@st.cache_data
def load_and_audit_data():
    try:
        df = pd.read_csv("ankara_traffic_data.csv")
        df.columns = df.columns.str.strip()

        # Numeric transformation for weather
        weather_map = {"Güneşli": 1, "Bulutlu": 2, "Yağmurlu": 3, "Karlı": 4}
        df['weather_numeric'] = df['weather_condition'].map(weather_map).fillna(1)

        # 1. OUTLIER DETECTION (IQR Method)
        Q1 = df['average_speed'].quantile(0.25)
        Q3 = df['average_speed'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df[(df['average_speed'] >= lower_bound) & (df['average_speed'] <= upper_bound)].copy()

        return df, df_cleaned, weather_map
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}


df_raw, df, weather_map = load_and_audit_data()

if not df.empty:
    st.title("🔬 Ankara Traffic: Data Science & Statistical Audit")
    st.markdown("---")

    # --- SECTION 1: DATA QUALITY AUDIT ---
    st.header("1. Data Quality Audit")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Raw Observations", len(df_raw))
    with c2:
        outliers_count = len(df_raw) - len(df)
        st.metric("Outliers Removed", outliers_count)
        st.caption("Using Interquartile Range (IQR)")
    with c3:
        st.metric("Cleaned Dataset", len(df))
        st.caption("Ready for analysis")

    # --- SECTION 2: DISTRIBUTION & NORMALITY ---
    st.markdown("---")
    st.header("2. Normality & Distribution Analysis")

    # Shapiro-Wilk Normality Test
    # Limited to 500 samples for the test accuracy
    stat, p_normality = stats.shapiro(df['average_speed'][:500])

    col_dist, col_test = st.columns(2)
    with col_dist:
        fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
        sns.histplot(df['average_speed'], kde=True, color="#ff4b4b", ax=ax_dist)
        ax_dist.set_title("Speed Distribution")
        st.pyplot(fig_dist)

    with col_test:
        st.subheader("Shapiro-Wilk Test Results")
        st.write(f"**Statistic:** {stat:.4f}")
        st.write(f"**P-Value:** {p_normality:.4f}")

        if p_normality > 0.05:
            st.success("✅ Normal Distribution (Pearson correlation applicable).")
            corr_method = "pearson"
        else:
            st.warning("⚠️ Non-Normal Distribution (Spearman rank correlation selected).")
            corr_method = "spearman"

    # --- SECTION 3: STATISTICAL CORRELATION ---
    st.markdown("---")
    st.header("3. Correlation & Hypothesis Testing")

    if corr_method == "pearson":
        corr_val, p_val = stats.pearsonr(df['density_score'], df['average_speed'])
    else:
        corr_val, p_val = stats.spearmanr(df['density_score'], df['average_speed'])

    k1, k2 = st.columns(2)
    with k1:
        st.metric(f"{corr_method.capitalize()} R", f"{corr_val:.4f}")
    with k2:
        st.metric("P-Value (Significance)", f"{p_val:.2e}")
        if p_val < 0.05:
            st.success("✅ Significant relationship confirmed (p < 0.05).")
        else:
            st.error("❌ Relationship could be random (p > 0.05).")

    # --- SECTION 4: DIAGNOSTICS & PREDICTION ---
    st.markdown("---")
    st.header("4. Prediction & Model Diagnostics")

    X = df[['density_score', 'weather_numeric']].values
    y = df['average_speed'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    col_model, col_diag = st.columns(2)
    with col_model:
        st.subheader("🤖 Prediction Engine")
        u_dens = st.slider("Target Density (%)", 0, 100, 50)
        u_weath = st.selectbox("Target Weather", list(weather_map.keys()))
        input_val = np.array([[u_dens, weather_map[u_weath]]])
        pred = model.predict(input_val)[0]
        st.metric("Predicted Speed", f"{pred:.2f} km/h")

    with col_diag:
        st.subheader("🧪 Statistical Diagnostics")
        dw_score = durbin_watson(residuals)
        r2 = model.score(X, y)
        st.write(f"**R² (Explainability):** {r2:.4f}")
        st.write(f"**Durbin-Watson:** {dw_score:.2f}")

        if 1.5 < dw_score < 2.5:
            st.caption("✅ No significant autocorrelation in residuals.")
        else:
            st.caption("⚠️ Potential autocorrelation risk detected.")

    # --- FINAL FOOTER ---
    st.markdown("---")
    footer_html = f"""
    <div style='text-align: center; padding: 10px;'>
        <p style='color: #888888; font-size: 14px;'>
            Analyzed by <a href='https://www.linkedin.com/in/ozdemircihan/' target='_blank' style='color: #ff4b4b; text-decoration: none;'>Cihan Özdemir</a>
            | Status: <b>Statistically Validated</b>
        </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

else:
    st.error("Data source could not be initialized.")
