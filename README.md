# SPK Klasifikasi DBD - ICD-10 A90 & A91

Sistem Pendukung Keputusan Klinis (Clinical Decision Support System) berbasis Streamlit untuk klasifikasi pasien Demam Berdarah Dengue (DBD) berdasarkan kode diagnosis ICD-10 A90 dan A91.

Project ini dikembangkan untuk kebutuhan penelitian/skripsi dengan pendekatan CRISP-DM dan model Decision Tree. Aplikasi ini berfungsi sebagai alat bantu keputusan klinis, bukan pengganti diagnosis dokter.

## Fitur utama

- Prediksi klasifikasi pasien DBD berdasarkan fitur laboratorium
- Input data pasien secara interaktif di web dashboard
- Output hasil prediksi, probabilitas, dan confidence score
- Visualisasi evaluasi model
- Interpretasi klinis untuk klasifikasi A90 dan A91
- Antarmuka dengan desain yang lebih rapi dan mudah dipahami

## Teknologi yang digunakan

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Matplotlib
- imbalanced-learn

## Struktur proyek

```text
PROJECT SKRIPSI SPK ICD/
├── README.md
├── .gitignore
├── .vscode/
├── docs/
│   └── PRD_DEPLOYMENT.md
├── notebooks/
│   └── SPK_Klasifikasi_DBD_Kode_ICD_RS_Aulia.ipynb
├── data/
│   └── raw/
│       └── Data_Lab_Penyakit_DBD_RS_Aulia.xlsx
├── scripts/
│   ├── eda/
│   │   ├── check_data.py
│   │   ├── get_sample.py
│   │   └── notebook_script.py
│   └── plots/
│       ├── plot_hasil_model.py
│       ├── plot_tree.py
│       ├── plot_tree_ilustrasi.py
│       └── plot_tree_light.py
├── outputs/
│   └── plots/
│       ├── decision_tree_light.png
│       └── hasil_model_lengkap.png
├── dbd_clinical_decision_support/
│   ├── app.py
│   ├── requirements.txt
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── notebook_train_model_dbd.ipynb
│   ├── assets/
│   │   ├── __init__.py
│   │   └── style.css
│   ├── dataset/
│   ├── models/
│   │   ├── pipeline_dbd.pkl
│   │   └── eval_metrics.pkl
│   ├── pages/
│   │   ├── 1_Prediksi.py
│   │   ├── 2_Evaluasi_Model.py
│   │   └── 3_Tentang_Sistem.py
│   └── .streamlit/
└── .git/
```

> Catatan: Folder utama aplikasi tetap berada di `dbd_clinical_decision_support/`. Sementara file penelitian, dataset, notebook, dan script eksplorasi dipindahkan ke folder yang lebih rapi agar repositori GitHub terlihat lebih profesional.

## Persiapan lingkungan

1. Clone repository ini.
2. Masuk ke folder project.
3. Buat virtual environment (opsional tapi disarankan).
4. Install dependency:

```bash
cd "dbd_clinical_decision_support"
pip install -r requirements.txt
```

## Menjalankan aplikasi

```bash
cd "dbd_clinical_decision_support"
streamlit run app.py
```

Setelah aplikasi berjalan, buka alamat lokal yang ditampilkan oleh Streamlit di browser.

## Melatih ulang model

```bash
cd "dbd_clinical_decision_support"
python train_model.py
```

Script ini akan membaca dataset, melatih model Decision Tree, dan menyimpan hasil model serta metrik evaluasi ke folder `models/`.

## Target klasifikasi

- A90 = Dengue Fever
- A91 = Dengue Hemorrhagic Fever

## Catatan penting

- Aplikasi ini merupakan sistem pendukung keputusan klinis.
- Hasil prediksi bukan keputusan medis final.
- Keputusan akhir tetap ditentukan oleh tenaga medis yang berwenang.

## Lisensi

Belum ditentukan secara eksplisit. Silakan sesuaikan lisensi sesuai kebutuhan institusi atau pembimbing jika project ini akan dipublikasikan atau dikembangkan lebih lanjut.

## Penanggung jawab / pengembang

Project ini dibuat sebagai bagian dari skripsi / tugas akhir terkait klasifikasi ICD-10 DBD menggunakan Decision Tree.
