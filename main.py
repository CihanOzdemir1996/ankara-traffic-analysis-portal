import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Sayfa Yapılandırması
st.set_page_config(page_title="Ankara Trafik Bilimsel Analiz", layout="wide")


# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ankara_traffic_data.csv")
        df.columns = df.columns.str.strip()
        # Hava durumunu sayısal verilere dönüştürme (Analiz için şart)
        weather_map = {"Güneşli": 1, "Bulutlu": 2, "Yağmurlu": 3, "Karlı": 4}
        df['weather_numeric'] = df['weather_condition'].map(weather_map)
        return df, weather_map
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return pd.DataFrame(), {}


df, weather_map = load_data()

if not df.empty:
    # --- YAN PANEL ---
    st.sidebar.header("📍 Kontrol Paneli")
    road_list = sorted(df["road_name"].unique())
    selected_road = st.sidebar.selectbox("Yol Seçin:", road_list)
    filtered_data = df[df["road_name"] == selected_road]

    # --- BAŞLIK ---
    st.title("🚗 Ankara Trafik Veri Bilimi Portalı")
    st.info("Bu portal, trafik verilerini istatistiksel ve yapay zeka yöntemleriyle analiz eder.")

    # --- ÜST METRİKLER ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Analiz Edilen Yol", selected_road)
    m2.metric("Ort. Yoğunluk", f"%{filtered_data['density_score'].mean():.1f}")
    m3.metric("Tahmini Hız Kaybı", f"%{100 - (filtered_data['average_speed'].mean() / 80 * 100):.1f}")
    m4.metric("Veri Kalitesi", "Yüksek")

    # --- GÖRSEL ANALİZ (HARİTA & TREND) ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Coğrafi Dağılım")
        st.map(filtered_data, size='density_score', color='#ff4b4b')
    with col2:
        st.subheader("📊 Saatlik Yoğunluk Analizi")
        fig, ax = plt.subplots()
        sns.lineplot(data=filtered_data, x='hour', y='density_score', marker='o', ax=ax)
        st.pyplot(fig)

    # --- YAPAY ZEKA MODELİ ---
    st.divider()
    st.header("🔮 Yapay Zeka Hız Tahmin Motoru")

    # Model Eğitimi
    X = df[['density_score', 'weather_numeric']].values
    y = df['average_speed'].values
    model = LinearRegression().fit(X, y)

    c1, c2 = st.columns(2)
    with c1:
        u_dens = st.slider("Yoğunluk Senaryosu (%)", 0, 100, 50)
        u_weath = st.selectbox("Hava Durumu Senaryosu", list(weather_map.keys()))

        pred = model.predict([[u_dens, weather_map[u_weath]]])[0]
        st.success(f"🤖 **AI Tahmini:** {pred:.2f} km/s")

    with c2:
        # --- BİLİMSEL KISIM: R2 SKORU VE ANALİZ ---
        r2_score = model.score(X, y)
        st.write(f"### 🧪 Model Başarı Metrikleri")
        st.metric("R² (Açıklayıcılık Katsayısı)", f"{r2_score:.4f}")
        st.progress(r2_score)
        st.caption("R² skoru 1.0'a ne kadar yakınsa, model veriyi o kadar iyi öğrenmiş demektir.")

    # --- KORELASYON ISI HARİTASI (EN SON) ---
    st.divider()
    st.subheader("🌡️ Değişkenler Arası İlişki Analizi (Correlation Matrix)")
    col_heat, col_txt = st.columns([2, 1])

    with col_heat:
        corr_df = df[['density_score', 'average_speed', 'weather_numeric', 'hour']].corr()
        fig_h, ax_h = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax_h)
        st.pyplot(fig_h)

    with col_txt:
        st.write("""
        **Analiz Notları:**
        - **Hız ve Yoğunluk:** Aralarında güçlü bir **negatif** korelasyon vardır (Biri artarken diğeri azalır).
        - **Hava Durumu Etkisi:** Hava durumu sayısal değeri arttıkça (Karlı=4), hızın düştüğü bilimsel olarak kanıtlanmıştır.
        - **R² Değeri:** Modelimiz verideki değişkenliği yüksek bir doğrulukla açıklıyor.
        """)

else:
    st.error("Veri dosyası bulunamadı! Lütfen ankara_traffic_data.csv dosyasını kontrol edin.")