# Buatkan aplikasi web Sistem Pendukung Keputusan Klinis untuk Klasifikasi Pasien DBD Berdasarkan Kode ICD-10 A90 dan A91

Buatkan aplikasi web **Sistem Pendukung Keputusan Klinis (Clinical Decision Support System)** berbasis **Streamlit** yang digunakan untuk membantu tenaga medis mengklasifikasikan pasien Demam Berdarah Dengue (DBD) berdasarkan kode diagnosis **ICD-10 A90 (Dengue Fever)** dan **ICD-10 A91 (Dengue Hemorrhagic Fever)**.

Aplikasi ini merupakan implementasi tahap **Deployment** dari penelitian menggunakan metodologi **CRISP-DM** dengan algoritma **Decision Tree**.

Website hanya berfungsi sebagai **Clinical Decision Support System (CDSS)** dan **bukan** sebagai sistem rekam medis.

---

# METODOLOGI PENELITIAN

Gunakan alur penelitian sesuai metode **CRISP-DM**:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

Seluruh implementasi harus mengikuti alur penelitian tersebut.

---

# TUJUAN SISTEM

Website harus mampu:

1. Menerima input data laboratorium pasien.
2. Melakukan preprocessing otomatis.
3. Menjalankan model Decision Tree.
4. Menghasilkan klasifikasi ICD-10.
5. Menampilkan confidence score.
6. Menampilkan probabilitas prediksi.
7. Menampilkan feature importance.
8. Menampilkan hasil evaluasi model.
9. Menampilkan interpretasi klinis.

---

# DATASET

Gunakan dataset:

```
Data_Lab_Penyakit_DBD_RS_Aulia.xlsx
```

Dataset hanya digunakan pada proses training model.

Website **tidak membaca dataset ketika dijalankan**.

Website hanya menggunakan file model hasil training.

---

# TEKNOLOGI

## Framework Deployment

- Streamlit

## Machine Learning

- Scikit-Learn
- Pandas
- NumPy
- Joblib

## Visualisasi

- Matplotlib
- Plotly

## Library Pendukung

- streamlit-option-menu

---

# MODEL MACHINE LEARNING

Gunakan model hasil penelitian.

```
pipeline_dbd.pkl
```

Pipeline telah mencakup:

- preprocessing
- encoding
- Decision Tree
- parameter terbaik hasil GridSearchCV
- class_weight sebagai metode penanganan ketidakseimbangan kelas

Jika pipeline tidak tersedia gunakan:

```
decision_tree.pkl
```

---

# FITUR INPUT

Gunakan HANYA variabel penelitian berikut:

- Usia
- Jenis Kelamin
- Trombosit
- Hematokrit
- Hemoglobin
- Leukosit

Jangan menambahkan fitur lain.

---

# PREPROCESSING

Lakukan preprocessing yang sama seperti penelitian.

Tahapan preprocessing meliputi:

- Validasi input
- Penanganan missing value apabila diperlukan
- Standardisasi format data
- Encoding Jenis Kelamin
- Penyusunan fitur sesuai urutan model

## Validasi

- Semua field wajib diisi.
- Semua input numerik bernilai positif.

## Encoding

Jenis Kelamin

- Laki-laki = 1
- Perempuan = 0

## Urutan Fitur

```python
[
    usia,
    jenis_kelamin,
    trombosit,
    hematokrit,
    hemoglobin,
    leukosit
]
```

Jika tersedia gunakan:

```
pipeline_dbd.pkl
```

agar preprocessing berjalan otomatis sesuai penelitian.

---

# TARGET KLASIFIKASI

Model menghasilkan dua kelas.

| Label | ICD | Keterangan |
|-------|-----|------------|
| 0 | A90 | Dengue Fever |
| 1 | A91 | Dengue Hemorrhagic Fever |

---

# HASIL PREDIKSI

Tampilkan:

- Hasil Prediksi
- ICD
- Confidence Score
- Probabilitas Prediksi
- Interpretasi Klinis
- Feature Importance

---

# INTERPRETASI KLINIS

Jika hasil:

## A90

Tampilkan:

> Dengue Fever (Risiko Klinis Lebih Rendah)

Jika hasil:

## A91

Tampilkan:

> Dengue Hemorrhagic Fever (Risiko Klinis Lebih Tinggi)

---

# FEATURE IMPORTANCE

Jika model memiliki atribut:

```
feature_importances_
```

Tampilkan:

- Ranking variabel
- Progress Bar
- Bar Chart

Urutkan dari nilai terbesar ke terkecil.

---

# EVALUASI MODEL

Tampilkan:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Tambahkan penjelasan bahwa penelitian memprioritaskan **Recall kelas A91** karena kesalahan False Negative pada kelas A91 dapat menyebabkan pasien dengan kondisi lebih berat diprediksi sebagai A90.

---

# ARSITEKTUR SISTEM

Gunakan arsitektur tiga lapis (Three Layer Architecture).

Presentation Layer

↓

Dashboard Streamlit

↓

Form Prediksi

↓

Application Layer

↓

Validasi Input

↓

Pipeline Preprocessing

↓

Machine Learning Layer

↓

Decision Tree

↓

Predict

↓

Predict Probability

↓

Feature Importance

↓

Hasil Prediksi

---

# HALAMAN WEBSITE

## Dashboard

Berisi:

- Judul penelitian
- Ringkasan penelitian
- Penjelasan metode CRISP-DM
- Penjelasan algoritma Decision Tree
- Informasi model
- Tombol Mulai Prediksi

---

## Prediksi Pasien

Form input:

- Usia
- Jenis Kelamin
- Trombosit
- Hematokrit
- Hemoglobin
- Leukosit

Button:

```
Prediksi
```

---

## Hasil Prediksi

Menampilkan:

- ICD
- Confidence
- Probability
- Interpretasi Klinis
- Feature Importance

Button:

```
Prediksi Lagi
```

---

## Evaluasi Model

Menampilkan:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## Tentang Sistem

Berisi:

- Tujuan penelitian
- Penjelasan CRISP-DM
- Cara kerja sistem
- Penjelasan Decision Tree
- Penjelasan Feature Importance

---

# DESAIN

Gunakan antarmuka modern berbasis Streamlit.

Tampilan harus:

- Responsif
- Bersih
- Profesional
- Dominan warna putih dan biru
- Sidebar Navigation
- Card UI
- Metric Card
- Tabs
- Expander
- Progress Bar
- Plotly Chart
- Ikon medis

Website harus mudah digunakan oleh tenaga medis.

---

# STRUKTUR PROYEK

```plaintext
dbd_clinical_decision_support/
│
├── app.py
├── train_model.py
├── preprocessing.py
├── pipeline_dbd.pkl
├── requirements.txt
│
├── dataset/
│   └── Data_Lab_Penyakit_DBD_RS_Aulia.xlsx
│
├── pages/
│   ├── 1_Prediksi.py
│   ├── 2_Evaluasi_Model.py
│   └── 3_Tentang_Sistem.py
│
├── assets/
│   ├── style.css
│   └── logo.png
│
└── models/
    └── pipeline_dbd.pkl
```

---

# NAVIGASI STREAMLIT

Navigasi aplikasi terdiri dari:

- Dashboard
- Prediksi Pasien
- Evaluasi Model
- Tentang Sistem

Navigasi menggunakan Sidebar atau `streamlit-option-menu`.

---

# KOMPONEN SISTEM

1. Modul Dashboard
2. Modul Input Data Pasien
3. Modul Validasi Data
4. Modul Preprocessing
5. Modul Prediksi Decision Tree
6. Modul Feature Importance
7. Modul Evaluasi Model
8. Modul Visualisasi Hasil

---

# WORKFLOW SISTEM

Input Data Pasien

↓

Validasi Input

↓

Pipeline Preprocessing

↓

Decision Tree

↓

Predict

↓

Predict Probability

↓

Interpretasi Klinis

↓

Feature Importance

↓

Tampilkan Hasil

---

# PENANGANAN ERROR

Jika file model:

```
pipeline_dbd.pkl
```

tidak ditemukan tampilkan pesan:

> Model belum tersedia. Silakan lakukan proses training terlebih dahulu.

Jika input tidak valid:

- Berikan notifikasi kesalahan.
- Jangan menjalankan prediksi.

---

# OUTPUT YANG DIMINTA

Buatkan secara lengkap:

1. app.py
2. train_model.py
3. preprocessing.py
4. requirements.txt
5. Folder pages
6. Folder assets
7. Visualisasi Feature Importance
8. Visualisasi Confusion Matrix
9. Penjelasan implementasi CRISP-DM
10. Penjelasan implementasi Decision Tree
11. Penjelasan arsitektur sistem
12. Contoh alur penggunaan sistem
13. Contoh hasil prediksi
14. Contoh tampilan aplikasi
15. Integrasi model pipeline_dbd.pkl

Pastikan seluruh kode:

- Menggunakan best practice Streamlit.
- Memiliki komentar yang jelas.
- Mudah dipahami.
- Siap dijalankan menggunakan:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```