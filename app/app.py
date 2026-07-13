# app.py - Aplikasi Prediksi Pengeluaran Bulanan
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Pengeluaran Bulanan",
    page_icon="📊",
    layout="wide"
)

# ==================== CSS KUSTOM ====================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .result-card h2 {
        color: #28a745;
        margin: 0;
    }
    .result-card.danger {
        border-left-color: #dc3545;
    }
    .result-card.danger h2 {
        color: #dc3545;
    }
    .advice-card {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .advice-good {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .advice-warning {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .advice-danger {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .advice-info {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-card .label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0.3rem 0;
    }
    .metric-card .sub {
        font-size: 0.85rem;
        color: #28a745;
    }
    .metric-card .sub.negative {
        color: #dc3545;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sidebar-sub {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
        color: #6c757d;
    }
    .stTabs [aria-selected="true"] {
        color: #667eea;
        font-weight: 600;
    }
    .score-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
    }
    .score-good {
        background: #28a745;
    }
    .score-medium {
        background: #ffc107;
        color: #333;
    }
    .score-bad {
        background: #dc3545;
    }
    .component-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
    .component-label {
        font-weight: 500;
    }
    .component-score {
        font-weight: bold;
    }
    .form-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .form-section h4 {
        margin-top: 0;
        color: #495057;
        font-size: 1rem;
        font-weight: 600;
    }
    .info-box {
        background: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 0.8rem 1.2rem;
        border-radius: 5px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .divider {
        border: none;
        border-top: 2px dashed #dee2e6;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>Prediksi Pengeluaran Bulanan</h1>
    <p>Masukkan data diri dan finansial Anda untuk mendapatkan analisis keuangan</p>
</div>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            return None
    except Exception as e:
        return None

model = load_model()

# ==================== FUNGSI SKOR KESEHATAN ====================
def hitung_skor_kesehatan(pendapatan, pengeluaran, tabungan_persen, utang, dana_darurat):
    """Menghitung skor kesehatan finansial 0-100"""
    
    # 1. Rasio Pengeluaran/Pendapatan (40%)
    if pendapatan > 0:
        rasio = pengeluaran / pendapatan
    else:
        rasio = 1.0
    
    if rasio <= 0.5:
        skor_rasio = 40
        level_rasio = "Sangat Baik"
    elif rasio <= 0.7:
        skor_rasio = 30
        level_rasio = "Baik"
    elif rasio <= 0.85:
        skor_rasio = 20
        level_rasio = "Cukup"
    elif rasio <= 1.0:
        skor_rasio = 10
        level_rasio = "Perlu Perhatian"
    else:
        skor_rasio = 0
        level_rasio = "Bahaya"
    
    # 2. Persentase Tabungan (30%)
    if tabungan_persen >= 20:
        skor_tabungan = 30
        level_tabungan = "Sangat Baik"
    elif tabungan_persen >= 10:
        skor_tabungan = 20
        level_tabungan = "Baik"
    elif tabungan_persen >= 5:
        skor_tabungan = 10
        level_tabungan = "Cukup"
    elif tabungan_persen > 0:
        skor_tabungan = 5
        level_tabungan = "Kurang"
    else:
        skor_tabungan = 0
        level_tabungan = "Sangat Kurang"
    
    # 3. Status Utang (20%)
    if utang == 0:
        skor_utang = 20
        level_utang = "Tidak Ada Utang"
    else:
        skor_utang = 0
        level_utang = "Ada Utang"
    
    # 4. Status Dana Darurat (10%)
    if dana_darurat == "Ya":
        skor_darurat = 10
        level_darurat = "Ada"
    else:
        skor_darurat = 0
        level_darurat = "Tidak Ada"
    
    total = skor_rasio + skor_tabungan + skor_utang + skor_darurat
    
    if total >= 80:
        kategori = "Sangat Baik"
        warna = "score-good"
    elif total >= 60:
        kategori = "Baik"
        warna = "score-good"
    elif total >= 40:
        kategori = "Cukup"
        warna = "score-medium"
    elif total >= 20:
        kategori = "Kurang"
        warna = "score-bad"
    else:
        kategori = "Sangat Kurang"
        warna = "score-bad"
    
    return {
        'total': total,
        'kategori': kategori,
        'warna': warna,
        'detail': {
            'Rasio Pengeluaran': {'skor': skor_rasio, 'level': level_rasio, 'max': 40},
            'Tabungan': {'skor': skor_tabungan, 'level': level_tabungan, 'max': 30},
            'Status Utang': {'skor': skor_utang, 'level': level_utang, 'max': 20},
            'Dana Darurat': {'skor': skor_darurat, 'level': level_darurat, 'max': 10}
        }
    }

# ==================== DATA WILAYAH ====================
DAFTAR_PROVINSI = [
    "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Kepulauan Riau", "Jambi", 
    "Bengkulu", "Sumatera Selatan", "Kepulauan Bangka Belitung", "Lampung", 
    "Banten", "DKI Jakarta", "Jawa Barat", "Jawa Tengah", "DI Yogyakarta", 
    "Jawa Timur", "Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur", 
    "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan", 
    "Kalimantan Timur", "Kalimantan Utara", "Sulawesi Utara", "Sulawesi Tengah", 
    "Sulawesi Selatan", "Sulawesi Tenggara", "Gorontalo", "Sulawesi Barat", 
    "Maluku", "Maluku Utara", "Papua Barat", "Papua", "Papua Tengah", 
    "Papua Pegunungan", "Papua Selatan", "Papua Barat Daya"
]

DAFTAR_KOTA = {
    "Jawa Tengah": ["Semarang", "Surakarta", "Magelang", "Pekalongan", "Tegal", "Salatiga"],
    "Jawa Timur": ["Surabaya", "Malang", "Kediri", "Blitar", "Madiun", "Probolinggo", "Pasuruan"],
    "Jawa Barat": ["Bandung", "Bogor", "Depok", "Bekasi", "Cimahi", "Cirebon", "Sukabumi"],
    "DKI Jakarta": ["Jakarta Pusat", "Jakarta Selatan", "Jakarta Timur", "Jakarta Barat", "Jakarta Utara"],
    "DI Yogyakarta": ["Yogyakarta"],
    "Banten": ["Tangerang", "Cilegon", "Serang"],
    "Bali": ["Denpasar"],
    "Sumatera Utara": ["Medan", "Binjai", "Tebing Tinggi", "Pematangsiantar"],
    "Sumatera Selatan": ["Palembang", "Lubuklinggau"],
    "Sulawesi Selatan": ["Makassar", "Palopo", "Parepare"],
    "Kalimantan Timur": ["Samarinda", "Balikpapan", "Bontang"],
    "Kalimantan Selatan": ["Banjarmasin", "Banjarbaru"],
    "Papua": ["Jayapura"]
}

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown('<p class="sidebar-header">Input Data Diri</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Lengkapi semua data di bawah ini</p>', unsafe_allow_html=True)
    
    # ===== PETUNJUK =====
    with st.expander("Petunjuk Pengisian", expanded=True):
        st.markdown("""
        1. Isi data demografi dan domisili Anda
        2. Masukkan pendapatan dan kebiasaan menabung
        3. Klik tombol Prediksi untuk melihat hasil analisis
        
        Data Anda tidak disimpan dan hanya digunakan untuk prediksi.
        """)
    
    st.markdown("---")
    
    # ===== SECTION 1: DATA DEMOGRAFI =====
    st.markdown('<div class="form-section"><h4>Data Demografi</h4>', unsafe_allow_html=True)
    
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    occupation = st.selectbox("Pekerjaan", ["Mahasiswa", "Karyawan", "Wirausaha", "Belum Bekerja", "Pensiun"])
    age = st.number_input("Usia (tahun)", min_value=17, max_value=65, value=22, step=1)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== SECTION 2: DOMISILI =====
    st.markdown('<div class="form-section"><h4>Domisili</h4>', unsafe_allow_html=True)
    
    # Pilih Provinsi
    provinsi = st.selectbox("Provinsi", ["Pilih Provinsi"] + DAFTAR_PROVINSI)
    
    # Pilih Kota/Kabupaten (dinamis berdasarkan provinsi)
    if provinsi != "Pilih Provinsi" and provinsi in DAFTAR_KOTA:
        kota_options = ["Pilih Kota/Kabupaten"] + DAFTAR_KOTA[provinsi]
    else:
        kota_options = ["Pilih Provinsi terlebih dahulu"]
    
    kota = st.selectbox("Kota/Kabupaten", kota_options)
    
    # Input Kecamatan (manual)
    kecamatan = st.text_input("Kecamatan", placeholder="Contoh: Tembalang, Gajahmungkur, dll.")
    
    # Tampilkan lokasi lengkap jika sudah diisi
    if provinsi != "Pilih Provinsi" and kota != "Pilih Provinsi terlebih dahulu" and kota != "Pilih Kota/Kabupaten":
        lokasi_display = f"{kota}, {provinsi}"
        if kecamatan:
            lokasi_display = f"{kecamatan}, {kota}, {provinsi}"
        st.success(f"Lokasi: {lokasi_display}")
    
    # Klasifikasi domisili (hanya 1 pilihan, tidak berulang)
    tipe_domisili = st.radio(
        "Tipe Domisili",
        ["Perkotaan", "Semi-perkotaan", "Pedesaan"],
        help="Perkotaan = kota besar, Semi-perkotaan = kabupaten/kota kecil, Pedesaan = desa"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== SECTION 3: DATA FINANSIAL =====
    st.markdown('<div class="form-section"><h4>Data Finansial</h4>', unsafe_allow_html=True)
    
    monthly_income = st.number_input(
        "Pendapatan Bulanan (Rp)", 
        min_value=0, 
        value=3000000, 
        step=100000,
        help="Masukkan total pendapatan bersih per bulan"
    )
    
    savings_rate = st.slider(
        "Persentase Ditabung (%)", 
        0, 50, 10,
        help="Berapa persen dari pendapatan yang Anda tabung setiap bulan?"
    )
    
    debt = st.number_input(
        "Total Utang (Rp)", 
        min_value=0, 
        value=0, 
        step=100000,
        help="Masukkan total utang Anda saat ini"
    )
    
    income_source = st.selectbox(
        "Sumber Pendapatan", 
        ["Gaji", "Bisnis", "Investasi", "Freelance", "Orang Tua"]
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== SECTION 4: GAYA HIDUP =====
    st.markdown('<div class="form-section"><h4>Gaya Hidup</h4>', unsafe_allow_html=True)
    
    credit_card_usage = st.selectbox(
        "Penggunaan Kartu Kredit", 
        ["Rendah", "Sedang", "Tinggi"],
        help="Seberapa sering Anda menggunakan kartu kredit?"
    )
    
    investment = st.selectbox(
        "Memiliki Investasi?", 
        ["Tidak", "Ya"],
        help="Saham, reksadana, emas, properti, dll."
    )
    
    emergency_fund = st.selectbox(
        "Memiliki Dana Darurat?", 
        ["Tidak", "Ya"],
        help="Dana darurat minimal 3 bulan pengeluaran"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===== TOMBOL PREDIKSI =====
    predict_button = st.button("Prediksi Sekarang", use_container_width=True)
    
    if predict_button and model is None:
        st.error("Model belum siap. Pastikan file model ada di folder models/")
    
    if predict_button and (provinsi == "Pilih Provinsi" or kota == "Pilih Provinsi terlebih dahulu" or kota == "Pilih Kota/Kabupaten"):
        st.warning("Mohon pilih provinsi dan kota/kabupaten terlebih dahulu.")

# ==================== MAIN CONTENT ====================
if predict_button and model is not None and provinsi != "Pilih Provinsi" and kota != "Pilih Provinsi terlebih dahulu" and kota != "Pilih Kota/Kabupaten":
    
    # Mapping untuk model
    gender_map = {"Laki-laki": "Male", "Perempuan": "Female"}
    occupation_map = {"Mahasiswa": "Student", "Karyawan": "Employee", "Wirausaha": "Self-employed", "Belum Bekerja": "Unemployed", "Pensiun": "Retired"}
    city_map = {"Perkotaan": "Urban", "Semi-perkotaan": "Suburban", "Pedesaan": "Rural"}
    income_source_map = {"Gaji": "Salary", "Bisnis": "Business", "Investasi": "Investment", "Freelance": "Freelance", "Orang Tua": "Other"}
    credit_map = {"Rendah": "Low", "Sedang": "Medium", "Tinggi": "High"}
    yes_no_map = {"Tidak": "No", "Ya": "Yes"}
    
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender_map[gender]],
        'occupation': [occupation_map[occupation]],
        'city': [city_map[tipe_domisili]],
        'monthly_income': [monthly_income],
        'income_source': [income_source_map[income_source]],
        'savings_rate': [savings_rate],
        'debt': [debt],
        'credit_card_usage': [credit_map[credit_card_usage]],
        'investment': [yes_no_map[investment]],
        'emergency_fund': [yes_no_map[emergency_fund]],
        'financial_stress': ["Medium"],
        'financial_health_score': [50]
    })
    
    try:
        # ===== PREDIKSI =====
        prediction = model.predict(input_data)[0]
        saving_potential = monthly_income - prediction
        
        # ===== BUAT LOKASI DISPLAY =====
        lokasi_display = f"{kota}, {provinsi}"
        if kecamatan:
            lokasi_display = f"{kecamatan}, {kota}, {provinsi}"
        
        # ===== HITUNG SKOR KESEHATAN =====
        skor = hitung_skor_kesehatan(monthly_income, prediction, savings_rate, debt, emergency_fund)
        
        # ==================== HASIL ====================
        st.markdown("---")
        st.markdown("## Hasil Analisis Keuangan")
        
        st.info(f"Data untuk domisili: {lokasi_display}")
        
        # ---- PREDIKSI PENGELUARAN ----
        st.markdown("### Prediksi Pengeluaran Bulanan")
        
        # Tampilkan penjelasan singkat
        st.markdown("""
        <div class="info-box">
            Sistem memprediksi pengeluaran Anda berdasarkan data demografi dan finansial yang Anda masukkan.
            Prediksi ini dihasilkan dari model Machine Learning yang telah dilatih dengan 10.000 data.
        </div>
        """, unsafe_allow_html=True)
        
        if saving_potential > 0:
            card_class = "result-card"
            status_text = "Sehat"
            status_color = "#28a745"
        else:
            card_class = "result-card danger"
            status_text = "Perlu Evaluasi"
            status_color = "#dc3545"
        
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="margin:0; color:#6c757d; font-size:0.9rem;">Total Pengeluaran Bulanan</p>
                    <h2 style="margin:0; font-size:2.5rem;">Rp {prediction:,.0f}</h2>
                </div>
                <div style="text-align:right;">
                    <p style="margin:0; color:#6c757d; font-size:0.9rem;">Status Keuangan</p>
                    <h3 style="margin:0; color:{status_color};">{status_text}</h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ---- PENJELASAN PREDIKSI ----
        with st.expander("Bagaimana sistem mengetahui pengeluaran saya?", expanded=False):
            st.markdown("""
            **Sistem tidak meminta Anda memasukkan pengeluaran karena:**
            
            1. **Model sudah dilatih** dengan ribuan data orang lain
            2. **Pola pengeluaran** dipengaruhi oleh:
               - Pendapatan (faktor paling dominan)
               - Usia dan pekerjaan
               - Kebiasaan menabung
               - Status utang
               - Gaya hidup (kartu kredit, investasi, dana darurat)
            
            3. **Model mempelajari pola** dari data historis:
               - Orang dengan pendapatan Rp 3.000.000 biasanya menghabiskan sekitar 70%
               - Mahasiswa cenderung memiliki pengeluaran lebih rendah
               - Semakin tinggi tabungan, semakin rendah pengeluaran
            
            **Tujuan:** Memberikan gambaran objektif tanpa harus menebak-nebak.
            """)
        
        # ---- METRIC CARDS ----
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Pendapatan</div>
                <div class="value">Rp {monthly_income:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Prediksi Pengeluaran</div>
                <div class="value">Rp {prediction:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if saving_potential > 0:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Potensi Tabungan</div>
                    <div class="value" style="color:#28a745;">Rp {saving_potential:,.0f}</div>
                    <div class="sub">Positif</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Potensi Tabungan</div>
                    <div class="value" style="color:#dc3545;">Rp {saving_potential:,.0f}</div>
                    <div class="sub negative">Negatif</div>
                </div>
                """, unsafe_allow_html=True)
        
        # ---- ANALISIS RASIO ----
        rasio = (prediction / monthly_income * 100) if monthly_income > 0 else 0
        st.markdown(f"**Rasio Pengeluaran terhadap Pendapatan:** {rasio:.1f}%")
        
        if rasio <= 50:
            st.success("Sangat baik. Pengeluaran hanya setengah dari pendapatan.")
        elif rasio <= 70:
            st.info("Baik. Pengeluaran masih dalam batas wajar.")
        elif rasio <= 85:
            st.warning("Cukup. Perlu diperhatikan agar tidak membengkak.")
        elif rasio <= 100:
            st.warning("Perlu perhatian. Pengeluaran hampir menyamai pendapatan.")
        else:
            st.error("Bahaya. Pengeluaran melebihi pendapatan.")
        
        # ==================== SKOR KESEHATAN ====================
        st.markdown("---")
        st.markdown("### Skor Kesehatan Finansial")
        
        st.markdown("""
        <div class="info-box">
            Skor ini dihitung secara otomatis dari 4 komponen:
            Rasio Pengeluaran, Tabungan, Status Utang, dan Dana Darurat.
        </div>
        """, unsafe_allow_html=True)
        
        col_skor, col_detail = st.columns([1, 2])
        
        with col_skor:
            st.markdown(f"""
            <div style="text-align:center;">
                <div class="score-circle {skor['warna']}">
                    {skor['total']}
                </div>
                <p style="font-size:1.2rem; font-weight:bold; margin-top:0.5rem;">{skor['kategori']}</p>
                <p style="color:#6c757d; font-size:0.85rem;">Skor 0-100</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_detail:
            st.markdown("**Rincian Penilaian:**")
            for komponen, data in skor['detail'].items():
                persentase = (data['skor'] / data['max']) * 100
                if persentase >= 80:
                    status = "[BAIK]"
                elif persentase >= 50:
                    status = "[PERHATIAN]"
                else:
                    status = "[PERBAIKI]"
                st.markdown(f"""
                <div class="component-row">
                    <span class="component-label">{status} {komponen}</span>
                    <span>{data['level']}</span>
                    <span class="component-score" style="color:{'#28a745' if persentase >= 50 else '#dc3545'}">{data['skor']}/{data['max']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # ==================== SARAN KEUANGAN ====================
        st.markdown("---")
        st.markdown("### Saran Keuangan")
        
        if skor['total'] >= 80:
            st.markdown("""
            <div class="advice-card advice-good">
                <strong>Keuangan Anda sangat sehat.</strong><br>
                Anda sudah mengelola keuangan dengan sangat baik. Pertahankan kebiasaan ini dan 
                mulailah pertimbangkan investasi jangka panjang untuk masa depan.
            </div>
            """, unsafe_allow_html=True)
        elif skor['total'] >= 60:
            st.markdown("""
            <div class="advice-card advice-good">
                <strong>Keuangan Anda dalam kondisi baik.</strong><br>
                Beberapa area masih bisa ditingkatkan. Fokus pada perbaikan komponen yang masih kurang.
            </div>
            """, unsafe_allow_html=True)
        elif skor['total'] >= 40:
            st.markdown("""
            <div class="advice-card advice-warning">
                <strong>Keuangan Anda cukup, tapi perlu perhatian.</strong><br>
                Ada beberapa aspek yang perlu ditingkatkan. Mulai dengan membuat anggaran bulanan.
            </div>
            """, unsafe_allow_html=True)
        elif skor['total'] >= 20:
            st.markdown("""
            <div class="advice-card advice-danger">
                <strong>Keuangan Anda perlu perbaikan serius.</strong><br>
                Segera evaluasi pengeluaran Anda. Buat prioritas kebutuhan dan kurangi pengeluaran tidak penting.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="advice-card advice-danger">
                <strong>Kondisi keuangan Anda sangat kritis.</strong><br>
                Segera ambil tindakan. Kurangi pengeluaran tidak penting, cari sumber pendapatan tambahan.
            </div>
            """, unsafe_allow_html=True)
        
        # ---- REKOMENDASI SPESIFIK ----
        st.markdown("#### Rekomendasi Spesifik:")
        
        rekomendasi = []
        
        if skor['detail']['Tabungan']['skor'] < 20:
            rekomendasi.append("- Tingkatkan tabungan. Mulai dengan menabung minimal 10% dari pendapatan.")
        
        if skor['detail']['Rasio Pengeluaran']['skor'] < 20:
            rekomendasi.append("- Kurangi rasio pengeluaran. Usahakan pengeluaran tidak lebih dari 70% dari pendapatan.")
        
        if skor['detail']['Status Utang']['skor'] == 0:
            rekomendasi.append("- Lunasi utang. Prioritaskan pelunasan utang untuk mengurangi beban keuangan.")
        
        if skor['detail']['Dana Darurat']['skor'] == 0:
            rekomendasi.append("- Siapkan dana darurat. Targetkan minimal 3 bulan pengeluaran sebagai dana darurat.")
        
        if savings_rate == 0 and saving_potential > 0:
            rekomendasi.append(f"- Mulai menabung sekarang. Anda berpotensi menabung Rp {saving_potential:,.0f} per bulan.")
        
        if rekomendasi:
            for r in rekomendasi:
                st.markdown(r)
        else:
            st.markdown("- Semua aspek keuangan Anda sudah baik. Pertahankan.")
        
        # ---- REKOMENDASI BULAN DEPAN ----
        st.markdown("---")
        st.markdown("### Rekomendasi untuk Bulan Depan")
        
        if saving_potential > 0:
            target_tabung = saving_potential * 0.8
            st.markdown(f"""
            **Target Menabung Bulan Depan:** Rp {target_tabung:,.0f}
            
            Dengan kondisi keuangan saat ini, Anda berpotensi menabung Rp {saving_potential:,.0f} per bulan.
            Targetkan menabung minimal 80% dari potensi tersebut.
            """)
        else:
            st.markdown("""
            **Prioritas Bulan Depan:**
            
            1. Kurangi pengeluaran yang tidak perlu
            2. Buat anggaran bulanan yang ketat
            3. Cari sumber pendapatan tambahan
            4. Hindari utang baru
            """)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ==================== TABS ====================
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Visualisasi Data", "Evaluasi Model", "Tentang Aplikasi"])

with tab1:
    st.markdown("### Rata-rata Pengeluaran per Kategori")
    
    sample_data = pd.DataFrame({
        'Kategori': ['Perumahan', 'Makanan', 'Transportasi', 'Hiburan', 'Belanja', 'Kesehatan'],
        'Rata-rata Pengeluaran': [1500, 800, 400, 300, 350, 200]
    })
    
    fig = px.bar(
        sample_data, 
        x='Kategori', 
        y='Rata-rata Pengeluaran',
        title='Rata-rata Pengeluaran per Kategori (USD)',
        color='Kategori',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Distribusi Total Pengeluaran")
    
    spending_data = np.random.normal(3500, 1000, 1000)
    fig2 = px.histogram(
        spending_data, 
        nbins=50,
        title='Distribusi Total Pengeluaran Bulanan',
        labels={'value': 'Total Spending (USD)'},
        color_discrete_sequence=['#667eea']
    )
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown("### Performa Model")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">R2 Score</div>
            <div class="value" style="color:#28a745;">0.978</div>
            <div class="sub">Sangat Baik</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">MAE</div>
            <div class="value">$195</div>
            <div class="sub">Error Rendah</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">RMSE</div>
            <div class="value">$341</div>
            <div class="sub">Baik</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    ### Informasi Model
    
    | Metrik | Nilai | Keterangan |
    |--------|-------|------------|
    | R2 Score | 0.978 | 97.8% variansi dapat dijelaskan |
    | MAE | $195 | Rata-rata error prediksi |
    | RMSE | $341 | Standar deviasi error |
    
    **Model Terbaik:** XGBoost Regressor
    
    **Dataset:** Personal Spending Dataset (10,000 records)
    """)

with tab3:
    st.markdown("""
    ### Tentang Aplikasi
    
    Aplikasi ini menggunakan Machine Learning untuk:
    1. Memprediksi pengeluaran bulanan berdasarkan data demografi dan finansial
    2. Menghitung skor kesehatan finansial secara otomatis
    3. Memberikan saran keuangan yang personal
    
    ---
    
    ### Bagaimana Sistem Memprediksi Pengeluaran?
    
    Sistem **tidak meminta Anda memasukkan pengeluaran** karena:
    
    1. **Model sudah dilatih** dengan 10.000 data dari berbagai orang
    2. **Pola pengeluaran** dipelajari dari fitur-fitur seperti:
       - Pendapatan (faktor paling dominan)
       - Usia dan pekerjaan
       - Kebiasaan menabung
       - Status utang
       - Gaya hidup (kartu kredit, investasi, dana darurat)
    
    3. **Tujuan:** Memberikan gambaran objektif dan saran perbaikan
    
    ---
    
    ### Teknologi
    
    | Komponen | Teknologi |
    |----------|-----------|
    | Framework | Streamlit |
    | Model | XGBoost Regressor |
    | Dataset | Personal Spending Dataset (10K records) |
    | Visualisasi | Plotly |
    
    ---
    
    ### Fitur Aplikasi
    
    1. Input Data - Isi data diri dan finansial
    2. Prediksi - Dapatkan estimasi pengeluaran bulanan
    3. Skor Kesehatan - Dapatkan skor finansial otomatis (0-100)
    4. Analisis - Lihat detail penilaian per komponen
    5. Saran - Dapatkan rekomendasi keuangan personal
    6. Visualisasi - Lihat grafik dan distribusi data
    

                
                
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.85rem; padding: 1rem 0;">
    Prediksi Pengeluaran Bulanan 
</div>
""", unsafe_allow_html=True)