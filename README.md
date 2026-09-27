# Sistem Pendukung Keputusan Klinis untuk Klasifikasi DBD - ICD-10 A90 & A91

Sistem Pendukung Keputusan Klinis (Clinical Decision Support System) berbasis Streamlit untuk klasifikasi pasien Demam Berdarah Dengue (DBD) berdasarkan kode diagnosis ICD-10 A90 dan A91.



## Ringkasan Proyek

Penelitian ini membangun model klasifikasi pasien Demam Berdarah Dengue (DBD) ke dalam kode diagnosis ICD-10 A90 dan A91 berdasarkan data klinis dan laboratorium. Aplikasi berfungsi sebagai *Clinical Decision Support System* (CDSS), bukan pengganti keputusan tenaga medis.

| Item | Detail |
| --- | --- |
| Peneliti | Enrique Giovanni Battista Djou |
| Metodologi | CRISP-DM |
| Algoritma | Decision Tree; pemilihan model membandingkan SMOTE dan `class_weight` berdasarkan *macro recall* |
| Target klasifikasi | ICD-10 A90 (Dengue Fever) dan A91 (Dengue Hemorrhagic Fever) |
| Dataset | Data klinis dan laboratorium pasien DBD RS Aulia |
| Fitur | 6 fitur: usia, jenis kelamin, trombosit, hematokrit, hemoglobin, dan leukosit |

## Ketidakseimbangan Data

Distribusi jumlah pasien pada kelas ICD-10 A90 dan A91 tidak seimbang. Kondisi ini dapat membuat model lebih cenderung memprediksi kelas dengan jumlah sampel lebih banyak, sehingga kinerja pada kelas yang lebih sedikit perlu diperhatikan.

Untuk menangani hal tersebut, proses pelatihan membandingkan dua pendekatan pada model Decision Tree: oversampling kelas minoritas menggunakan SMOTE dan pemberian bobot kelas menggunakan `class_weight`. Data dibagi menjadi data latih dan data uji terlebih dahulu; SMOTE hanya diterapkan pada data latih agar data uji tetap terpisah dan tidak ikut digunakan dalam proses penyeimbangan. Model terbaik dipilih berdasarkan nilai *macro recall*, yang menghitung recall tiap kelas secara seimbang.

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

## Disclaimer

- Aplikasi ini merupakan sistem pendukung keputusan klinis.
- Hasil prediksi bukan keputusan medis final.
- Keputusan akhir tetap ditentukan oleh tenaga medis yang berwenang.

## Lisensi

Belum ditentukan secara eksplisit. Silakan sesuaikan lisensi sesuai kebutuhan institusi atau pembimbing jika project ini akan dipublikasikan atau dikembangkan lebih lanjut.

## Penanggung jawab / pengembang

Project ini dibuat sebagai bagian dari skripsi / tugas akhir terkait klasifikasi ICD-10 DBD menggunakan Decision Tree.

## Link

https://spk-klasifikasi-icd.streamlit.app
