import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Sayfa Yapılandırması
st.set_page_config(page_title="Ankara Trafik Dashboard V2", layout="wide")


# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ankara_traffic_data.csv")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return pd.DataFrame()


df = load_data()

# --- ARAYÜZ ÜST KISIM ---
st.title("🚗 Ankara Akıllı Trafik Yönetim Portalı")
st.markdown("Gerçek verilerle Ankara ana arter trafiği, harita analizi ve **Gelişmiş AI** hız tahmini.")

if not df.empty:
    # --- YAN PANEL (FİLTRE) ---
    st.sidebar.header("Yol ve Veri Seçimi")
    road_list = sorted(df["road_name"].unique())
    selected_road = st.sidebar.selectbox("Analiz Edilecek Yolu Seçin:", road_list)

    filtered_data = df[df["road_name"] == selected_road]

    # --- ÜST İSTATİSTİK KARTLARI ---
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Seçilen Yol", selected_road)
    with m2:
        st.metric("En Yüksek Yoğunluk", f"%{filtered_data['density_score'].max()}")
    with m3:
        st.metric("Ortalama Hız", f"{filtered_data['average_speed'].mean():.1f} km/s")
    with m4:
        st.metric("Kayıt Sayısı", f"{len(filtered_data)} Saat dilimi")

    # --- HARİTA VE GRAFİK ---
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("📍 Lokasyon Analizi")
        map_df = filtered_data[['lat', 'lon', 'density_score']].dropna()
        st.map(map_df, size='density_score', color='#ff4b4b')

    with col_right:
        st.subheader("📊 Saatlik Yoğunluk Trendi")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=filtered_data.sort_values('hour'), x="hour", y="density_score", palette="magma", ax=ax)
        st.pyplot(fig)

    # --- VERİ TABLOSU ---
    st.divider()
    st.subheader("📋 Detaylı Veri Kayıtları")
    st.dataframe(
        filtered_data.style.highlight_between(left=80, right=100, subset=['density_score'], color='#ff4b4b'),
        use_container_width=True
    )

    # --- GELİŞMİŞ YAPAY ZEKA (AI+) ---
    st.divider()
    st.header("🔮 Gelişmiş Hız Tahmincisi (AI+)")
    st.write("Bu model hem trafik yoğunluğunu hem de hava durumunu analiz ederek daha gerçekçi sonuçlar üretir.")

    # Hava durumunu sayısal verilere dönüştürme
    weather_map = {"Güneşli": 1, "Bulutlu": 2, "Yağmurlu": 3, "Karlı": 4}
    df['weather_numeric'] = df['weather_condition'].map(weather_map)

    # Model Eğitimi (Yoğunluk ve Hava Durumu -> Hız)
    X = df[['density_score', 'weather_numeric']].values
    y = df['average_speed'].values
    model_v2 = LinearRegression().fit(X, y)

    col_ai1, col_ai2 = st.columns(2)
    with col_ai1:
        user_density = st.slider("Tahmin için Yoğunluk Seviyesi (%)", 0, 100, 50)
    with col_ai2:
        user_weather = st.selectbox("Hava Durumu Senaryosu Seçin:", list(weather_map.keys()))

    # Tahmin yapma
    weather_val = weather_map[user_weather]
    prediction = model_v2.predict([[user_density, weather_val]])

    st.success(f"**{user_weather}** havada, **%{user_density}** yoğunlukta tahmini hız: **{prediction[0]:.2f} km/s**")

else:
    st.warning("Veri bulunamadı.")