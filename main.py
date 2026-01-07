import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Sayfa Yapılandırması
st.set_page_config(page_title="Ankara Trafik Dashboard", layout="wide")


# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ankara_traffic_data.csv")
        df.columns = df.columns.str.strip()  # Sütun isimlerindeki boşlukları temizle
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return pd.DataFrame()


df = load_data()

# --- ARAYÜZ ÜST KISIM ---
st.title("🚗 Ankara Akıllı Trafik Yönetim Portalı")
st.markdown("Gerçek verilerle Ankara ana arter trafiği, harita analizi ve AI hız tahmini.")

if not df.empty:
    # --- YAN PANEL (FİLTRE) ---
    st.sidebar.header("Yol ve Veri Seçimi")
    road_list = sorted(df["road_name"].unique())
    selected_road = st.sidebar.selectbox("Analiz Edilecek Yolu Seçin:", road_list)

    # Seçilen yola göre veriyi filtrele
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

    # --- HARİTA VE GRAFİK (YAN YANA) ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📍 Lokasyon Analizi")
        # Harita için koordinatları hazırlıyoruz
        map_df = filtered_data[['lat', 'lon', 'density_score']].dropna()
        st.map(map_df, size='density_score', color='#ff4b4b')
        st.caption("Kırmızı noktanın büyüklüğü trafik yoğunluğunu temsil eder.")

    with col_right:
        st.subheader("📊 Saatlik Yoğunluk Trendi")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=filtered_data.sort_values('hour'), x="hour", y="density_score", palette="magma", ax=ax)
        ax.set_ylabel("Yoğunluk (%)")
        ax.set_xlabel("Saat")
        st.pyplot(fig)

    # --- VERİ TABLOSU VE UYARILAR ---
    st.divider()
    st.subheader("📋 Detaylı Veri Kayıtları")
    st.info("Yoğunluğu %80'in üzerinde olan kritik saatler aşağıda vurgulanmıştır.")
    st.dataframe(
        filtered_data.style.highlight_between(left=80, right=100, subset=['density_score'], color='#ff4b4b'),
        use_container_width=True
    )

    # --- YAPAY ZEKA MODELİ ---
    st.divider()
    st.header("🔮 Yapay Zeka Hız Tahmincisi")
    st.write("Seçilen yolun geçmiş verileri kullanılarak makine öğrenmesi modeli eğitilmiştir.")

    # Model: Yoğunluk -> Hız (Tüm veriden eğitmek daha mantıklı)
    X = df[['density_score']].values
    y = df['average_speed'].values
    model = LinearRegression().fit(X, y)

    user_input = st.slider("İleride beklenen yoğunluk seviyesini seçin (%)", 0, 100, 50)
    prediction = model.predict([[user_input]])

    st.success(f"Bu yoğunluk seviyesinde beklenen ortalama hız: **{prediction[0]:.2f} km/s**")

else:
    st.warning("Veri seti bulunamadı. Lütfen CSV dosyasını kontrol edin.")
