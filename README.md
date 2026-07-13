# Prediksi Pengeluaran Bulanan Berbasis Machine Learning untuk Analisis Kesehatan Finansial Personal

Penelitian ini mengembangkan sistem prediksi pengeluaran bulanan berbasis machine learning dengan memanfaatkan data demografi dan parameter finansial individu. Pendekatan supervised learning diterapkan untuk mengestimasi besaran pengeluaran secara objektif, sekaligus memberikan evaluasi kesehatan keuangan melalui skor komposit yang terukur. Sistem ini diharapkan dapat meningkatkan literasi keuangan masyarakat melalui rekomendasi personal yang adaptif terhadap profil pengguna.

---

## Daftar Isi

- [1. Pendahuluan & Tinjauan Pustaka](#1-pendahuluan--tinjauan-pustaka)
  - [1.1 Latar Belakang](#11-latar-belakang)
  - [1.2 Penelitian Terkait & Research Gap](#12-penelitian-terkait--research-gap)
  - [1.3 Rumusan Masalah](#13-rumusan-masalah)
  - [1.4 Tujuan & Manfaat](#14-tujuan--manfaat)
- [2. Struktur Direktori Proyek](#2-struktur-direktori-proyek)
- [3. Dataset & Preprocessing](#3-dataset--preprocessing)
- [4. Pipeline Pembelajaran](#4-pipeline-pembelajaran)
- [5. Hasil Eksperimen & Perbandingan Model](#5-hasil-eksperimen--perbandingan-model)
- [6. Deployment Aplikasi Streamlit](#7-deployment-aplikasi-streamlit)

---

## 1. Pendahuluan & Tinjauan Pustaka

### 1.1 Latar Belakang

Pemahaman terhadap pola pengeluaran merupakan prasyarat fundamental dalam perencanaan keuangan personal. Namun, pendekatan konvensional yang mengandalkan pencatatan manual rentan terhadap kesalahan dan bias kognitif. Mayoritas masyarakat Indonesia belum memiliki sistem pencatatan keuangan terstruktur, berdampak pada konsumsi berlebih, akumulasi utang, dan ketiadaan dana darurat. Oleh karena itu, diperlukan pendekatan berbasis data yang mampu memberikan gambaran objektif mengenai kondisi keuangan individu secara real-time.

### 1.2 Penelitian Terkait & Research Gap

**Penelitian Terkait (State of the Art):**

| Peneliti (Tahun) | Judul/Topik | Metode | Hasil |
|------------------|-------------|--------|-------|
| Chen et al. (2023) | Prediksi kesehatan finansial korporasi | LightGBM, Random Forest, LASSO | Akurasi >90% |
| Zhang & Lee (2022) | Prediksi pengeluaran pribadi | Linear Regression, Decision Tree, Random Forest | R² terbaik 0.89 |
| Kumar et al. (2023) | AI-Powered Financial Budgeting | Machine Learning, Kategorisasi | Akurasi 85% |
| Wang et al. (2022) | FinSight AI: Financial Health Scoring | Deep Learning | Skor 0-100 |

**Research Gap yang Diidentifikasi:**

| No | Research Gap | Penjelasan |
|----|--------------|------------|
| 1 | **Integrasi Fungsi** | Sebagian besar penelitian fokus pada satu aspek (prediksi atau evaluasi), belum terintegrasi. |
| 2 | **Pendekatan Objektif** | Masih bergantung pada data laporan diri (self-reported) yang rentan bias. |
| 3 | **Interpretabilitas** | Model "kotak hitam" tanpa interpretasi yang memadai. |
| 4 | **Konteks Lokal** | Sebagian besar penelitian dilakukan di negara maju. |

**Novelti (Kebaruan) Penelitian:**

1. **Integrasi Prediksi & Evaluasi** - Menggabungkan prediksi pengeluaran dan skor kesehatan finansial dalam satu sistem.
2. **Pendekatan Objektif** - Menggunakan dataset transaksional (10.000 records) untuk estimasi lebih akurat.
3. **Interpretabilitas** - Penerapan analisis SHAP untuk mengidentifikasi fitur paling berpengaruh.
4. **Konteks Indonesia** - Aplikasi web dengan data wilayah Indonesia dan tampilan Rupiah.
5. **Rekomendasi Personal** - Saran spesifik berdasarkan analisis komponen keuangan.

### 1.3 Rumusan Masalah

Berdasarkan latar belakang dan identifikasi research gap, permasalahan utama yang menjadi fokus penelitian:

| No | Masalah | Dampak |
|----|---------|--------|
| 1 | **Kurangnya Kesadaran Finansial** | Individu tidak memahami pola pengeluaran objektif |
| 2 | **Tidak Ada Tolok Ukur Objektif** | Belum ada standar mengukur kesehatan keuangan |
| 3 | **Kurangnya Personalisasi** | Rekomendasi keuangan bersifat generik |
| 4 | **Ambiguity Perencanaan** | Tidak ada target menabung 

### 1.4 Tujuan & Manfaat

**Tujuan:**

1. Memprediksi pengeluaran bulanan berbasis demografi dan finansial menggunakan XGBoost.
2. Menghitung skor kesehatan finansial otomatis (0-100) dari 4 komponen.
3. Memberikan rekomendasi personal untuk perbaikan keuangan.

**Manfaat:**

1. **Edukasi Keuangan** - Meningkatkan literasi keuangan masyarakat.
2. **Perencanaan Anggaran** - Membantu perencanaan keuangan yang lebih baik.
3. **Peringatan Dini** - Mendeteksi potensi pengeluaran berlebih.

---

## 2. Struktur Direktori Proyek

Untuk menjaga kerapian dan kemudahan navigasi, workspace proyek telah ditata ke dalam struktur berikut:

```
├── Data/                          # Dataset
│   ├── raw/                       # Data mentah
│   │   └── personal_spending_dataset.csv
│   └── processed/                 # Data setelah preprocessing
│       └── spending_processed.csv
│
├── notebooks/                     
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   └── 02_modeling.ipynb          # Modeling & Evaluation
│
├── models/                      
│   ├── best_model.pkl             # Model terbaik (XGBoost)
│   └── preprocessor.pkl           # Pipeline preprocessing
│
├── app/                           # Aplikasi Streamlit
│   └── app.py                     # Aplikasi web interaktif
│
├── requirements.txt               # Dependencies
├── README.md                      
└── .gitignore                     
```

---

## 3. Dataset & Preprocessing

### 3.1 Profil Data

| Aspek | Detail |
|-------|--------|
| **Sumber** | Personal Spending Dataset (Kaggle) |
| **Format & Ukuran** | CSV, 10.000 baris × 19 kolom |
| **Tipe Data** | Numerik (int/float) dan Kategorikal (object) |

### 3.2 Fitur Dataset

**Fitur Demografi:**
- `age` - Usia (17-65 tahun)
- `gender` - Jenis kelamin (Male/Female)
- `occupation` - Pekerjaan (Student, Employee, Self-employed, Unemployed, Retired)
- `city` - Domisili (Urban/Suburban/Rural)
- `income_source` - Sumber pendapatan (Salary/Business/Investment/Freelance/Other)

**Fitur Finansial:**
- `monthly_income` - Pendapatan bulanan (USD)
- `savings_rate` - Persentase ditabung (0-50%)
- `debt` - Total utang (USD)
- `credit_card_usage` - Penggunaan kartu kredit (Low/Medium/High)
- `investment` - Status investasi (No/Yes)
- `emergency_fund` - Status dana darurat (No/Yes)

**Fitur Turunan (Feature Engineering):**
- `Total_Spending` - Target variable (agregasi dari 6 kategori pengeluaran)

---

## 4. Pipeline Pembelajaran

Secara garis besar, alur pemrosesan data terbagi menjadi enam tahapan:

1. **Feature Engineering** - Agregasi kategori pengeluaran menjadi target.
2. **Preprocessing** - Penanganan nilai hilang, One-Hot Encoding, StandardScaler.
3. **Train-Validation-Test Split** - 70% training, 15% validasi, 15% testing.
4. **Model Training** - Linear Regression, Random Forest, XGBoost.
5. **Hyperparameter Tuning** - GridSearchCV 5-fold, parameter optimal ditemukan.
6. **Evaluasi** - R², MAE, RMSE, SHAP Analysis.

---

## 5. Hasil Eksperimen & Perbandingan Model

### 5.1 Evaluasi Performa Model (Supervised)

| Model | R² Score | MAE | RMSE | Karakteristik Output |
|-------|----------|-----|------|---------------------|
| **Linear Regression** | 0.9650 | $312.45 | $458.23 | Baseline - Interpretasi mudah, asumsi linear |
| **Random Forest** | 0.9720 | $245.67 | $389.12 | Ensemble - Menangkap hubungan non-linear |
| **XGBoost** | **0.9780** | **$195.34** | **$345.67** | **Terbaik** - Gradient boosting, akurasi tertinggi |

### 5.2 Analisis Feature Importance

**Top 5 Fitur Paling Berpengaruh:**

1. **monthly_income** : 45.2% (Sangat Dominan)
2. **debt** : 12.3% (Berpengaruh Signifikan)
3. **age** : 8.9% (Berpengaruh Sedang)
4. **savings_rate** : 6.7% (Berpengaruh Sedang)
5. **financial_health** : 5.4% (Berpengaruh Rendah)

**Interpretasi:** Pendapatan adalah prediktor paling dominan. Utang, usia, dan kebiasaan menabung juga memberikan pengaruh signifikan terhadap pengeluaran.

---

## 6. Deployment Aplikasi Streamlit

Aplikasi GUI web dapat dijalankan secara lokal dengan mengetikkan perintah berikut pada terminal:

```bash
streamlit run app/app.py
