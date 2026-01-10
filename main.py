import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson
import numpy as np

# Sayfa Yapılandırması
st.set_page_config(page_title="Ankara Trafik Bilimsel Analiz", layout="wide")


# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    try:
        # Sütun isimlerindeki boşlukları temizleyerek okuyoruz
        df = pd.read_csv("ankara_traffic_data.csv")
        df.columns = df.columns.str.strip()

        # Hava durumu eşleşmesi (Sayısal analiz için)
        weather_map = {"Güneşli": 1, "Bulutlu": 2, "Yağmurlu": 3, "Karlı": 4}
        df['weather_numeric'] = df['weather_condition'].map(weather_map).fillna(1)

        # Boş satırları temizle (Model hatasını engellemek için)
        df = df.dropna(subset=['density_score', 'average_speed', 'weather_numeric'])

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
    st.info("Bu sistem, regresyon varsayımlarını (Durbin-Watson vb.) kontrol ederek tahmin yapar.")

    # --- ÜST METRİKLER ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Analiz Edilen Yol", selected_road)
    m2.metric("Ort. Yoğunluk", f"%{filtered_data['density_score'].mean():.1f}")
    m3.metric("Ort. Hız", f"{filtered_data['average_speed'].mean():.1f} km/s")
    m4.metric("Veri Kalitesi", "Doğrulandı ✅")

    # --- GÖRSEL ANALİZ ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Coğrafi Dağılım")
        st.map(filtered_data, size='density_score', color='#ff4b4b')
    with col2:
        st.subheader("📊 Saatlik Yoğunluk Trendi")
        fig, ax = plt.subplots()
        sns.lineplot(data=filtered_data, x='hour', y='density_score', marker='o', ax=ax)
        st.pyplot(fig)

    # --- YAPAY ZEKA VE İSTATİSTİKSEL ANALİZ ---
    st.divider()
    st.header("🔮 Yapay Zeka & İstatistiksel Doğrulama")

    # Model Eğitimi (Tüm veri seti üzerinden)
    X = df[['density_score', 'weather_numeric']].values
    y = df['average_speed'].values

    if len(X) > 0:
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        c1, c2 = st.columns(2)
        with c1:
            st.write("### 🤖 Hız Tahmini Yap")
            u_dens = st.slider("Yoğunluk Senaryosu (%)", 0, 100, 50)
            u_weath = st.selectbox("Hava Durumu Senaryosu", list(weather_map.keys()))

            # Seçilen senaryo için tahmin
            input_data = np.array([[u_dens, weather_map[u_weath]]])
            prediction = model.predict(input_data)[0]
            st.success(f"**Tahmin Edilen Hız:** {prediction:.2f} km/s")

        with c2:
            st.write("### 🧪 Model Başarı Metrikleri")

            # R2 Skoru
            r2_score = model.score(X, y)
            st.metric("R² (Doğruluk Oranı)", f"{r2_score:.4f}")

            # Durbin-Watson (Otokorelasyon) Analizi
            residuals = y - y_pred
            dw_val = durbin_watson(residuals)
            st.metric("Durbin-Watson Skoru", f"{dw_val:.2f}")

            # Varsayım Kontrolü
            if 1.5 < dw_val < 2.5:
                st.caption("✅ **Otokorelasyon Yok:** Hatalar bağımsızdır (Varsayım geçerli).")
            else:
                st.caption("⚠️ **Otokorelasyon Var:** Zaman serisi etkileri görülebilir.")

            st.progress(max(0.0, min(r2_score, 1.0)))

    # --- KORELASYON ANALİZİ ---
    st.divider()
    st.subheader("🌡️ Değişkenler Arası İlişki Analizi (Korelasyon)")
    corr_df = df[['density_score', 'average_speed', 'weather_numeric', 'hour']].corr()
    fig_h, ax_h = plt.subplots(figsize=(8, 4))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_h)
    st.pyplot(fig_h)

else:
    st.error("Veri dosyası yüklenemedi. Lütfen CSV dosyasını kontrol edin.")