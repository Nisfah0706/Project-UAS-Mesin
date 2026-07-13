# Prediksi Pengeluaran Bulanan: Pendekatan Supervised Learning untuk Analisis Keuangan Personal

Proyek ini bertujuan untuk memprediksi total pengeluaran bulanan seseorang berdasarkan karakteristik demografi dan finansial menggunakan pendekatan pembelajaran terstruktur (Supervised Learning Pipeline). Sistem ini dirancang untuk membantu individu memahami pola pengeluaran mereka, mengevaluasi kesehatan keuangan, dan mendapatkan rekomendasi personal untuk perbaikan keuangan di masa depan.

---

## Daftar Isi

- [1. Struktur Direktori Proyek](#1-struktur-direktori-proyek)
- [2. Formulasi Permasalahan & Ringkasan Akademik](#2-formulasi-permasalahan--ringkasan-akademik)
  - [2.1 Judul Jurnal Utama](#21-judul-jurnal-utama)
  - [2.2 Latar Belakang (Context)](#22-latar-belakang-context)
  - [2.3 Rumusan Masalah](#23-rumusan-masalah)
- [3. Dataset & Preprocessing](#3-dataset--preprocessing)
  - [3.1 Profil Data](#31-profil-data)
  - [3.2 Fitur Dataset](#32-fitur-dataset)
- [4. Pipeline Pembelajaran](#4-pipeline-pembelajaran)
- [5. Hasil Eksperimen & Perbandingan Model](#5-hasil-eksperimen--perbandingan-model)
  - [5.1 Evaluasi Performa Model (Supervised)](#51-evaluasi-performa-model-supervised)
  - [5.2 Analisis Feature Importance](#52-analisis-feature-importance)
- [6. Daftar Gambar Visualisasi Utama](#6-daftar-gambar-visualisasi-utama)
- [7. Deployment Aplikasi Streamlit](#7-deployment-aplikasi-streamlit)

---

## 1. Struktur Direktori Proyek

Untuk menjaga kerapian dan kemudahan navigasi, workspace proyek telah ditata ke dalam struktur berikut:

---

## 2. Formulasi Permasalahan & Ringkasan Akademik

### 2.1 Judul Jurnal Utama

**"Prediksi Pengeluaran Bulanan Berbasis Machine Learning: Pendekatan Supervised Learning untuk Analisis Keuangan Personal"**

### 2.2 Latar Belakang (Context)

Dalam literasi keuangan personal, pemahaman terhadap pola pengeluaran merupakan fondasi utama untuk perencanaan keuangan yang sehat. Pendekatan konvensional mengandalkan pencatatan manual yang rentan terhadap kesalahan dan bias subjektif. Berdasarkan data, lebih dari 60% masyarakat Indonesia tidak memiliki catatan pengeluaran bulanan yang terstruktur, yang berdampak pada:

- **Rendahnya tingkat literasi keuangan** - Masyarakat tidak memahami pola pengeluaran mereka secara objektif
- **Tingginya angka utang konsumtif** - Pengeluaran melebihi pendapatan tanpa disadari
- **Kurangnya perencanaan masa depan** - Tidak ada dana darurat atau investasi

Sistem prediksi pengeluaran berbasis machine learning hadir sebagai solusi dengan pendekatan objektif berbasis data. Sistem ini menggunakan model supervised learning yang telah dilatih dengan ribuan data historis untuk mengidentifikasi pola hubungan antara karakteristik demografi, perilaku finansial, dan besaran pengeluaran bulanan.

### 2.3 Rumusan Masalah

1. **Kurangnya Kesadaran Finansial** - Individu tidak mengetahui pola pengeluaran objektif, sehingga kesulitan merencanakan anggaran.
2. **Ketiadaan Alat Ukur Kesehatan Finansial** - Tidak ada standar untuk menilai kondisi keuangan secara objektif.
3. **Tidak Ada Rekomendasi Personal** - Saran keuangan bersifat umum dan tidak spesifik, sehingga perbaikan keuangan tidak terarah.
4. **Target Keuangan Tidak Jelas** - Tidak ada panduan menabung yang realistis dan terukur.

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

**Interpretasi:**
- Pendapatan adalah prediktor paling kuat untuk pengeluaran
- Utang dan usia juga sangat mempengaruhi pola pengeluaran
- Kebiasaan menabung mempengaruhi pengeluaran secara tidak langsung

---

## 6. Daftar Gambar Visualisasi Utama

Proyek ini menghasilkan visualisasi pembuktian yang disertakan dalam artikel ilmiah:

| Gambar | Deskripsi |
|--------|-----------|
| `spending_distribution.png` (Figure 1) | Histogram distribusi total pengeluaran bulanan |
| `income_vs_spending.png` (Figure 2) | Scatter plot hubungan pendapatan vs pengeluaran |
| `category_spending.png` (Figure 3) | Bar chart rata-rata pengeluaran per kategori |
| `gender_comparison.png` (Figure 4) | Boxplot perbandingan pengeluaran berdasarkan gender |
| `correlation_heatmap.png` (Figure 5) | Heatmap korelasi antar fitur |
| `model_comparison.png` (Figure 6) | Bar chart perbandingan performa 3 model |
| `prediction_scatter.png` (Figure 7) | Scatter plot prediksi vs aktual |
| `residual_plot.png` (Figure 8) | Residual plot untuk analisis error |
| `feature_importance.png` (Figure 9) | Feature importance Top 15 |
| `shap_summary.png` (Figure 10) | SHAP summary plot interpretasi |

---

## 7. Deployment Aplikasi Streamlit

Aplikasi GUI web dapat dijalankan secara lokal dengan mengetikkan perintah berikut pada terminal:

```bash
streamlit run app/app.py
