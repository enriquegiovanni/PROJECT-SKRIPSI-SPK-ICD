# Manipulasi dan analisis data
import pandas as pd
import numpy as np
import re

# Visualisasi data
import matplotlib.pyplot as plt
import seaborn as sns

# Statistik
from scipy import stats

# Preprocessing dan pemodelan
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score, ConfusionMatrixDisplay)

# Penanganan data tidak seimbang
from imblearn.over_sampling import SMOTE

# Penyimpanan model
import joblib

# Pengaturan tampilan
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

RANDOM_STATE = 42
print("Seluruh library berhasil diimpor.")

df_raw = pd.read_excel('Data_Lab_Penyakit_DBD_RS_Aulia.xlsx')
print(f"Jumlah baris  : {df_raw.shape[0]}")
print(f"Jumlah kolom  : {df_raw.shape[1]}")
df_raw.shape

# Lima data pertama
df_raw.head()

# Lima data terakhir
df_raw.tail()

# Nama seluruh kolom pada dataset
df_raw.columns.tolist()

# Tipe data masing-masing kolom
df_raw.dtypes

# Informasi umum dataset
df_raw.info()

# Statistik deskriptif dataset
df_raw.describe(include='all').T

target_counts = df_raw['kode ICD'].value_counts()
target_percent = df_raw['kode ICD'].value_counts(normalize=True) * 100

print("Jumlah data per kelas ICD:")
print(target_counts)
print("\nPersentase data per kelas ICD:")
print(target_percent.round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart
sns.barplot(x=target_counts.index, y=target_counts.values, hue=target_counts.index,
            palette=['#4C72B0', '#DD8452'], ax=axes[0], legend=False)
axes[0].set_title('Distribusi Jumlah Pasien Berdasarkan Kode ICD', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Kode ICD')
axes[0].set_ylabel('Jumlah Pasien')
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.2f%%',
            colors=['#4C72B0', '#DD8452'], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title('Proporsi Pasien Berdasarkan Kode ICD', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('distribusi_target.png', dpi=150, bbox_inches='tight')
plt.show()

kolom_dipilih = {
    'Usia (tahun)': 'Usia',
    'Jenis Kelamin\n(L/P)': 'Jenis_Kelamin',
    'Trombosit\n(ribu/μL)': 'Trombosit',
    'Hematokrit (%)': 'Hematokrit',
    'Hemoglobin (g/dL)': 'Hemoglobin',
    'Leukosit\n(ribu/μL)': 'Leukosit',
    'kode ICD': 'ICD'
}

df = df_raw[list(kolom_dipilih.keys())].rename(columns=kolom_dipilih).copy()
print(f"Dataset hasil seleksi variabel: {df.shape[0]} baris, {df.shape[1]} kolom")
df.head()

def parse_usia(teks):
    '''Mengekstraksi usia (dalam tahun) dari format teks campuran hasil rekam medis.'''
    if pd.isna(teks):
        return np.nan
    teks = str(teks).strip()
    if teks == '':
        return np.nan
    # Format dengan satuan tahun ("Th" / "tahun")
    cocok = re.search(r'(\d+)\s*(Th|tahun)', teks, re.IGNORECASE)
    if cocok:
        return float(cocok.group(1))
    # Pasien bayi (< 1 tahun), hanya tertulis dalam bulan/hari
    cocok = re.search(r'(\d+)\s*(Bl|bulan)', teks, re.IGNORECASE)
    if cocok:
        return 0.0
    return np.nan

def parse_nilai_lab(teks):
    '''Mengekstraksi nilai numerik dari format "nilai (rentang rujukan)".'''
    if pd.isna(teks):
        return np.nan
    teks = str(teks).strip()
    if teks == '':
        return np.nan
    cocok = re.search(r'([\d]+\.?[\d]*)', teks)
    return float(cocok.group(1)) if cocok else np.nan

df['Usia'] = df['Usia'].apply(parse_usia)
for kolom in ['Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']:
    df[kolom] = df[kolom].apply(parse_nilai_lab)

df['Jenis_Kelamin'] = df['Jenis_Kelamin'].astype(str).str.strip().str.upper()
df['ICD'] = df['ICD'].astype(str).str.strip().str.upper()

print("Konversi format data selesai.")
df.head()

df.dtypes

missing_jumlah = df.isnull().sum()
missing_persen = (df.isnull().sum() / len(df) * 100).round(2)

missing_table = pd.DataFrame({
    'Jumlah Missing Value': missing_jumlah,
    'Persentase (%)': missing_persen
})
missing_table

# Menandai baris yang seluruh nilai laboratoriumnya kosong (tidak menjalani
# pemeriksaan darah lengkap pada kunjungan tersebut). Penanda ini digunakan
# sebagai bahan diagnostik pada tahap pemeriksaan duplikat berikutnya.
kolom_lab = ['Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']
df['_tanpa_data_lab'] = df[kolom_lab].isnull().all(axis=1)

kolom_numerik = ['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']
kolom_kategorikal = ['Jenis_Kelamin']

imputer_numerik = SimpleImputer(strategy='median')
imputer_kategorikal = SimpleImputer(strategy='most_frequent')

df[kolom_numerik] = imputer_numerik.fit_transform(df[kolom_numerik])
df[kolom_kategorikal] = imputer_kategorikal.fit_transform(df[kolom_kategorikal])

print("Imputasi missing value selesai.")
print("\nJumlah missing value setelah imputasi:")
print(df.drop(columns=['_tanpa_data_lab']).isnull().sum())

jumlah_duplikat = df.duplicated(subset=['Usia','Jenis_Kelamin','Trombosit','Hematokrit','Hemoglobin','Leukosit','ICD']).sum()
print(f"Jumlah baris duplikat (berdasarkan kombinasi 6 fitur + target): {jumlah_duplikat}")
print(f"Persentase terhadap total data: {jumlah_duplikat / len(df) * 100:.2f}%")

# Diagnostik: seberapa besar duplikat di atas berasal dari baris yang
# sebelum imputasi tidak memiliki data laboratorium sama sekali (lihat
# penanda '_tanpa_data_lab' pada tahap Missing Value)
mask_duplikat = df.duplicated(subset=['Usia','Jenis_Kelamin','Trombosit','Hematokrit','Hemoglobin','Leukosit','ICD'], keep=False)
total_dalam_grup_duplikat = mask_duplikat.sum()
tanpa_lab_dalam_grup_duplikat = df.loc[mask_duplikat, '_tanpa_data_lab'].sum()

print(f"Total baris yang tergabung dalam kelompok duplikat : {total_dalam_grup_duplikat}")
print(f"Di antaranya, baris tanpa data laboratorium asli   : {tanpa_lab_dalam_grup_duplikat} "
      f"({tanpa_lab_dalam_grup_duplikat / total_dalam_grup_duplikat * 100:.2f}%)")

df = df.drop_duplicates(subset=['Usia','Jenis_Kelamin','Trombosit','Hematokrit','Hemoglobin','Leukosit','ICD']).reset_index(drop=True)
df = df.drop(columns=['_tanpa_data_lab'])

print(f"Jumlah data setelah penghapusan duplikat: {df.shape[0]} baris")

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
kolom_boxplot = ['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']

for i, kolom in enumerate(kolom_boxplot):
    sns.boxplot(y=df[kolom], ax=axes[i], color='#4C72B0')
    axes[i].set_title(kolom, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('')

plt.tight_layout()
plt.savefig('boxplot_outlier.png', dpi=150, bbox_inches='tight')
plt.show()

# Pemeriksaan nilai yang secara klinis tidak mungkin (nol atau negatif)
print("Jumlah nilai <= 0 pada tiap variabel numerik (indikasi kesalahan pencatatan):")
print((df[kolom_boxplot] <= 0).sum())

print("Nilai unik Jenis_Kelamin sebelum standardisasi:")
print(df['Jenis_Kelamin'].unique())

# Memastikan hanya terdapat kategori 'L' dan 'P'
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].replace({
    'LAKI-LAKI': 'L', 'LAKI LAKI': 'L', 'PRIA': 'L', 'M': 'L',
    'PEREMPUAN': 'P', 'WANITA': 'P', 'F': 'P'
})
df = df[df['Jenis_Kelamin'].isin(['L', 'P'])].reset_index(drop=True)

print("\nNilai unik Jenis_Kelamin setelah standardisasi:")
print(df['Jenis_Kelamin'].value_counts())

df['Jenis_Kelamin'] = df['Jenis_Kelamin'].map({'L': 1, 'P': 0})
df['ICD'] = df['ICD'].map({'A91': 1, 'A90': 0})

print("Hasil encoding:")
df.head()

X = df[['Usia', 'Jenis_Kelamin', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']]
y = df['ICD']

print(f"Bentuk X (fitur) : {X.shape}")
print(f"Bentuk y (target): {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"Jumlah data training : {X_train.shape[0]}")
print(f"Jumlah data testing  : {X_test.shape[0]}")

print("\nDistribusi kelas pada data training:")
print(y_train.value_counts())
print("\nDistribusi kelas pada data testing:")
print(y_test.value_counts())

print("Distribusi kelas sebelum SMOTE (data training):")
print(y_train.value_counts())

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nDistribusi kelas sesudah SMOTE (data training):")
print(y_train_smote.value_counts())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

sebelum = y_train.value_counts().sort_index()
sesudah = y_train_smote.value_counts().sort_index()
label_kelas = ['A90', 'A91']

sns.barplot(x=label_kelas, y=sebelum.values, hue=label_kelas, palette=['#4C72B0', '#DD8452'], ax=axes[0], legend=False)
axes[0].set_title('Distribusi Kelas Sebelum SMOTE', fontweight='bold')
axes[0].set_ylabel('Jumlah Data')
for i, v in enumerate(sebelum.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

sns.barplot(x=label_kelas, y=sesudah.values, hue=label_kelas, palette=['#4C72B0', '#DD8452'], ax=axes[1], legend=False)
axes[1].set_title('Distribusi Kelas Sesudah SMOTE', fontweight='bold')
axes[1].set_ylabel('Jumlah Data')
for i, v in enumerate(sesudah.values):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('smote_distribusi.png', dpi=150, bbox_inches='tight')
plt.show()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

dt_base = DecisionTreeClassifier(random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=dt_base,
    param_grid=param_grid,
    scoring='recall_macro',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_smote, y_train_smote)

print("Pencarian hyperparameter (GridSearchCV) selesai.")

print("Parameter terbaik:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")

print(f"\nBest Score (recall_macro, rata-rata cross-validation): {grid_search.best_score_:.4f}")

model_terbaik = grid_search.best_estimator_
print(f"\nModel terbaik: {model_terbaik}")

y_pred = model_terbaik.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['A90', 'A91'], digits=4))

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5.5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['A90', 'A91'])
disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
ax.set_title('Confusion Matrix - Data Testing', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negative  (A90 diprediksi A90): {tn}")
print(f"False Positive (A90 diprediksi A91): {fp}")
print(f"False Negative (A91 diprediksi A90): {fn}")
print(f"True Positive  (A91 diprediksi A91): {tp}")

akurasi = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

recall_a91 = recall_score(y_test, y_pred, pos_label=1)
precision_a91 = precision_score(y_test, y_pred, pos_label=1)
f1_a91 = f1_score(y_test, y_pred, pos_label=1)

ringkasan_metrik = pd.DataFrame({
    'Metrik': ['Accuracy', 'Precision (Macro Avg)', 'Recall (Macro Avg)', 'F1-Score (Macro Avg)',
               'Precision (Weighted Avg)', 'Recall (Weighted Avg)', 'F1-Score (Weighted Avg)',
               'Precision (A91)', 'Recall (A91)', 'F1-Score (A91)'],
    'Nilai': [akurasi, precision_macro, recall_macro, f1_macro,
              precision_weighted, recall_weighted, f1_weighted,
              precision_a91, recall_a91, f1_a91]
})
ringkasan_metrik['Nilai'] = ringkasan_metrik['Nilai'].round(4)
ringkasan_metrik

cv_scores = cross_val_score(model_terbaik, X_train_smote, y_train_smote,
                             cv=cv_strategy, scoring='recall_macro')

for i, skor in enumerate(cv_scores, 1):
    print(f"Fold {i}: {skor:.4f}")

print(f"\nMean Recall Macro     : {cv_scores.mean():.4f}")
print(f"Standar Deviasi        : {cv_scores.std():.4f}")

fitur_importance = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': model_terbaik.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

fitur_importance

plt.figure(figsize=(8, 5))
sns.barplot(data=fitur_importance, x='Importance', y='Fitur', hue='Fitur',
            palette='Blues_r', legend=False)
plt.title('Feature Importance - Decision Tree', fontweight='bold')
plt.xlabel('Tingkat Kepentingan Fitur')
plt.ylabel('')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

kedalaman_model = model_terbaik.get_depth()
jumlah_daun = model_terbaik.get_n_leaves()
print(f"Kedalaman pohon keputusan (full tree) : {kedalaman_model}")
print(f"Jumlah leaf node (full tree)          : {jumlah_daun}")

plt.figure(figsize=(28, 14))
plot_tree(model_terbaik, max_depth=3, feature_names=X.columns, class_names=['A90', 'A91'],
          filled=True, rounded=True, fontsize=10, proportion=False)
plt.title('Visualisasi Decision Tree (3 Tingkat Pertama)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()

train_pred = model_terbaik.predict(X_train_smote)
train_accuracy = accuracy_score(y_train_smote, train_pred)
test_accuracy = accuracy_score(y_test, y_pred)
gap = train_accuracy - test_accuracy

overfitting_table = pd.DataFrame({
    'Metrik': ['Training Accuracy', 'Testing Accuracy', 'Train-Test Gap'],
    'Nilai': [round(train_accuracy, 4), round(test_accuracy, 4), round(gap, 4)]
})
overfitting_table

param_grid_iterasi = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'class_weight': ['balanced', None]
}

# Catatan: GridSearchCV pada tahap iterasi ini dilatih langsung pada X_train, y_train
# (TANPA SMOTE), karena penyeimbangan kelas pada iterasi ini ditangani melalui
# parameter class_weight, bukan melalui penambahan sampel sintetis.
grid_search_iterasi = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid=param_grid_iterasi,
    scoring='recall_macro',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=0
)

grid_search_iterasi.fit(X_train, y_train)

print("Parameter terbaik (hasil iterasi):")
for k, v in grid_search_iterasi.best_params_.items():
    print(f"  {k}: {v}")
print(f"\nBest Score (recall_macro, cross-validation): {grid_search_iterasi.best_score_:.4f}")

model_iterasi = grid_search_iterasi.best_estimator_

y_pred_iterasi = model_iterasi.predict(X_test)

print("Classification Report - Model Hasil Iterasi:\n")
print(classification_report(y_test, y_pred_iterasi, target_names=['A90', 'A91'], digits=4))

cm_iterasi = confusion_matrix(y_test, y_pred_iterasi)

fig, ax = plt.subplots(figsize=(5.5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_iterasi, display_labels=['A90', 'A91'])
disp.plot(ax=ax, cmap='Greens', colorbar=False, values_format='d')
ax.set_title('Confusion Matrix - Model Hasil Iterasi', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_iterasi.png', dpi=150, bbox_inches='tight')
plt.show()

perbandingan_model = pd.DataFrame({
    'Metrik': ['Accuracy', 'Precision (A91)', 'Recall (A91)', 'F1-Score (A91)', 'Recall (Macro Avg)'],
    'Model Awal (SMOTE)': [
        round(accuracy_score(y_test, y_pred), 4),
        round(precision_score(y_test, y_pred, pos_label=1), 4),
        round(recall_score(y_test, y_pred, pos_label=1), 4),
        round(f1_score(y_test, y_pred, pos_label=1), 4),
        round(recall_score(y_test, y_pred, average='macro'), 4),
    ],
    'Model Hasil Iterasi (class_weight)': [
        round(accuracy_score(y_test, y_pred_iterasi), 4),
        round(precision_score(y_test, y_pred_iterasi, pos_label=1), 4),
        round(recall_score(y_test, y_pred_iterasi, pos_label=1), 4),
        round(f1_score(y_test, y_pred_iterasi, pos_label=1), 4),
        round(recall_score(y_test, y_pred_iterasi, average='macro'), 4),
    ]
})
perbandingan_model

# Pemilihan model final berdasarkan Recall Macro Average tertinggi pada data testing,
# dengan mempertimbangkan keseimbangan Precision-Recall pada kelas A91
recall_macro_awal = recall_score(y_test, y_pred, average='macro')
recall_macro_iterasi = recall_score(y_test, y_pred_iterasi, average='macro')

if recall_macro_iterasi > recall_macro_awal:
    model_final = model_iterasi
    nama_model_final = "Model Hasil Iterasi (class_weight='balanced')"
    y_pred_final = y_pred_iterasi
else:
    model_final = model_terbaik
    nama_model_final = "Model Awal (SMOTE)"
    y_pred_final = y_pred

print(f"Model final yang dipilih: {nama_model_final}")
print(f"Recall Macro Average pada data testing: {recall_score(y_test, y_pred_final, average='macro'):.4f}")

from sklearn.base import clone

preprocessor = ColumnTransformer(
    transformers=[
        ('imputer_numerik', SimpleImputer(strategy='median'), ['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']),
        ('imputer_kategorikal', SimpleImputer(strategy='most_frequent'), ['Jenis_Kelamin'])
    ],
    remainder='drop'
)

pipeline_dbd = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', clone(model_final))
])

# Data latih yang digunakan untuk melatih ulang pipeline disesuaikan dengan
# proses pelatihan model_final: data hasil SMOTE apabila model final adalah
# Model Awal, atau data training asli apabila model final adalah Model Hasil Iterasi.
if nama_model_final.startswith("Model Awal"):
    pipeline_dbd.fit(X_train_smote, y_train_smote)
else:
    pipeline_dbd.fit(X_train, y_train)

pipeline_pred = pipeline_dbd.predict(X_test)
print(f"Model final yang digunakan pada pipeline: {nama_model_final}")
print(f"Akurasi pipeline pada data testing: {accuracy_score(y_test, pipeline_pred):.4f}")
print("(Nilai ini digunakan untuk memverifikasi bahwa pipeline final menghasilkan performa yang konsisten dengan model_final pada Tahap Evaluation/Iterasi.)")

joblib.dump(pipeline_dbd, 'pipeline_dbd.pkl')
joblib.dump(model_final, 'model_decision_tree.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Model dan pipeline berhasil disimpan:")
print("  - pipeline_dbd.pkl        (pipeline lengkap: preprocessing + model)")
print(f"  - model_decision_tree.pkl (model final: {nama_model_final})")
print("  - preprocessor.pkl        (tahap preprocessing/imputasi saja)")

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Dataset berhasil disimpan:")
print("  - X_train.csv")
print("  - X_test.csv")
print("  - y_train.csv")
print("  - y_test.csv")

