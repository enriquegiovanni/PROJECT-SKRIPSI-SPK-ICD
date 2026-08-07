"""
train_model.py
==============
Script Training Model Decision Tree untuk CDSS DBD
Menerapkan logika EXACT dari Jupyter Notebook penelitian:
- SimpleImputer (median & most_frequent)
- SMOTE (imblearn)
- 2x GridSearchCV (SMOTE vs class_weight)
- Memilih model terbaik berdasarkan Recall Macro
- Export Pipeline dengan ColumnTransformer
"""

import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
import joblib

# Reconfigure stdout for emoji support on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Imbalanced Learn
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  Konfigurasi Path
# ─────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH  = os.path.join(BASE_DIR, "dataset", "Data_Lab_Penyakit_DBD_RS_Aulia.xlsx")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "pipeline_dbd.pkl")
METRICS_PATH  = os.path.join(MODEL_DIR, "eval_metrics.pkl")
RANDOM_STATE  = 42

os.makedirs(MODEL_DIR, exist_ok=True)


def print_section(title: str):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# ─────────────────────────────────────────────
#  Fungsi Parsing (sama persis dengan notebook)
# ─────────────────────────────────────────────

def parse_usia(teks):
    if pd.isna(teks): return np.nan
    teks = str(teks).strip()
    if teks == '': return np.nan
    cocok = re.search(r'(\d+)\s*(Th|tahun)', teks, re.IGNORECASE)
    if cocok: return float(cocok.group(1))
    cocok = re.search(r'(\d+)\s*(Bl|bulan)', teks, re.IGNORECASE)
    if cocok: return 0.0
    return np.nan

def parse_nilai_lab(teks):
    if pd.isna(teks): return np.nan
    teks = str(teks).strip()
    if teks == '': return np.nan
    cocok = re.search(r'([\d]+\.?[\d]*)', teks)
    return float(cocok.group(1)) if cocok else np.nan

# ─────────────────────────────────────────────
#  MAIN SCRIPT
# ─────────────────────────────────────────────

def main():
    print_section("FASE 1: MEMUAT DATASET")
    df_raw = pd.read_excel(DATASET_PATH)
    print(f"Dataset dimuat: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom.")

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

    print_section("FASE 2: PARSING & IMPUTASI")
    # Parsing teks
    df['Usia'] = df['Usia'].apply(parse_usia)
    for kolom in ['Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']:
        df[kolom] = df[kolom].apply(parse_nilai_lab)
    
    df['Jenis_Kelamin'] = df['Jenis_Kelamin'].astype(str).str.strip().str.upper()
    df['ICD'] = df['ICD'].astype(str).str.strip().str.upper()

    # Penanda tanpa lab
    kolom_lab = ['Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']
    df['_tanpa_data_lab'] = df[kolom_lab].isnull().all(axis=1)

    # Imputasi Missing Values
    kolom_numerik = ['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']
    kolom_kategorikal = ['Jenis_Kelamin']

    imputer_numerik = SimpleImputer(strategy='median')
    imputer_kategorikal = SimpleImputer(strategy='most_frequent')

    df[kolom_numerik] = imputer_numerik.fit_transform(df[kolom_numerik])
    df[kolom_kategorikal] = imputer_kategorikal.fit_transform(df[kolom_kategorikal])

    # Hapus duplikat
    df = df.drop_duplicates(subset=['Usia','Jenis_Kelamin','Trombosit','Hematokrit','Hemoglobin','Leukosit','ICD']).reset_index(drop=True)
    df = df.drop(columns=['_tanpa_data_lab'])
    print(f"Dataset setelah imputasi dan hapus duplikat: {df.shape[0]} baris.")

    # Standardisasi Kategorikal
    df['Jenis_Kelamin'] = df['Jenis_Kelamin'].replace({
        'LAKI-LAKI': 'L', 'LAKI LAKI': 'L', 'PRIA': 'L', 'M': 'L',
        'PEREMPUAN': 'P', 'WANITA': 'P', 'F': 'P'
    })
    df = df[df['Jenis_Kelamin'].isin(['L', 'P'])].reset_index(drop=True)

    # Encoding
    df['Jenis_Kelamin'] = df['Jenis_Kelamin'].map({'L': 1, 'P': 0})
    df['ICD'] = df['ICD'].map({'A91': 1, 'A90': 0})

    X = df[['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit', 'Jenis_Kelamin']] # Susunan fitur diubah karena Jenis_Kelamin akan diproses ColumnTransformer di akhir
    y = df['ICD'].astype(int)

    print_section("FASE 3: TRAIN TEST SPLIT & SMOTE")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Data SMOTE: {y_train_smote.value_counts().to_dict()}")

    print_section("FASE 4: MODELING (1) - SMOTE")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    param_grid_smote = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search_smote = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid_smote,
        scoring='recall_macro',
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0
    )
    grid_search_smote.fit(X_train_smote, y_train_smote)
    model_smote = grid_search_smote.best_estimator_
    y_pred_smote = model_smote.predict(X_test)
    recall_macro_smote = recall_score(y_test, y_pred_smote, average='macro')
    print(f"Recall Macro SMOTE: {recall_macro_smote:.4f}")

    print_section("FASE 4: MODELING (2) - CLASS WEIGHT (Iterasi)")
    param_grid_iterasi = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'class_weight': ['balanced', None]
    }
    grid_search_iterasi = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid_iterasi,
        scoring='recall_macro',
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0
    )
    grid_search_iterasi.fit(X_train, y_train)
    model_iterasi = grid_search_iterasi.best_estimator_
    y_pred_iterasi = model_iterasi.predict(X_test)
    recall_macro_iterasi = recall_score(y_test, y_pred_iterasi, average='macro')
    print(f"Recall Macro Iterasi (class_weight): {recall_macro_iterasi:.4f}")

    print_section("FASE 5: PEMILIHAN MODEL FINAL")
    if recall_macro_iterasi > recall_macro_smote:
        model_final = model_iterasi
        nama_model_final = "Model Hasil Iterasi (class_weight='balanced')"
        y_pred_final = y_pred_iterasi
    else:
        model_final = model_smote
        nama_model_final = "Model Awal (SMOTE)"
        y_pred_final = y_pred_smote
    print(f"🏆 Model Terpilih: {nama_model_final}")

    # Build Pipeline dengan ColumnTransformer
    # Catatan: Karena di Jupyter, preprocessor ditaruh di dalam pipeline, 
    # maka pipeline ini mengharapkan DataFrame dgn kolom: 'Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit', 'Jenis_Kelamin'
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

    # Fit ulang pipeline
    if nama_model_final.startswith("Model Awal"):
        pipeline_dbd.fit(X_train_smote, y_train_smote)
    else:
        pipeline_dbd.fit(X_train, y_train)

    print_section("FASE 6: EVALUASI & EXPORT")
    
    # Kalkulasi Metrics
    accuracy = accuracy_score(y_test, y_pred_final)
    precision = precision_score(y_test, y_pred_final, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred_final, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred_final, average='macro', zero_division=0)

    precision_pc = precision_score(y_test, y_pred_final, average=None, zero_division=0)
    recall_pc = recall_score(y_test, y_pred_final, average=None, zero_division=0)
    f1_pc = f1_score(y_test, y_pred_final, average=None, zero_division=0)

    cm = confusion_matrix(y_test, y_pred_final)
    cr = classification_report(y_test, y_pred_final, target_names=['A90', 'A91'], output_dict=True)
    cr_text = classification_report(y_test, y_pred_final, target_names=['A90', 'A91'])

    # Feature Importance (dari model final yg diclone lalu difit ulang)
    dt_model = pipeline_dbd.named_steps['classifier']
    feature_importances = dt_model.feature_importances_
    
    # Urutan fitur output dari ColumnTransformer:
    # 5 numerik, lalu 1 kategorik.
    feature_names = ['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit', 'Jenis_Kelamin']

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'train_accuracy': float(accuracy_score(
            y_train_smote if nama_model_final.startswith("Model Awal") else y_train, 
            pipeline_dbd.predict(X_train_smote if nama_model_final.startswith("Model Awal") else X_train)
        )),
        'precision_a90': float(precision_pc[0]),
        'recall_a90': float(recall_pc[0]),
        'f1_a90': float(f1_pc[0]),
        'precision_a91': float(precision_pc[1]),
        'recall_a91': float(recall_pc[1]),
        'f1_a91': float(f1_pc[1]),
        'confusion_matrix': cm.tolist(),
        'classification_report': cr,
        'classification_report_text': cr_text,
        'feature_importances': feature_importances.tolist(),
        'feature_names': feature_names,
        'class_counts': {
            'A90': int((y_test == 0).sum()),
            'A91': int((y_test == 1).sum()),
        },
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_total': len(X_train) + len(X_test),
        'best_model_name': nama_model_final
    }

    joblib.dump(pipeline_dbd, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    print(f"✅ Pipeline dan Metrik berhasil disimpan.")
    print(f"   Accuracy: {accuracy*100:.2f}% | Recall A91: {recall_pc[1]*100:.2f}%")

if __name__ == "__main__":
    main()
