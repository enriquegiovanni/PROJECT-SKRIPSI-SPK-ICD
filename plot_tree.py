"""
plot_tree.py
============
Script untuk menghasilkan visualisasi Decision Tree model DBD
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

DATASET_PATH = r"Data_Lab_Penyakit_DBD_RS_Aulia.xlsx"
OUTPUT_PATH  = r"decision_tree_dbd.png"
RANDOM_STATE = 42

# ── Parsing ───────────────────────────────────────────────────
def parse_usia(teks):
    if pd.isna(teks): return np.nan
    teks = str(teks).strip()
    cocok = re.search(r'(\d+)\s*(Th|tahun)', teks, re.IGNORECASE)
    if cocok: return float(cocok.group(1))
    cocok = re.search(r'(\d+)\s*(Bl|bulan)', teks, re.IGNORECASE)
    if cocok: return 0.0
    return np.nan

def parse_nilai_lab(teks):
    if pd.isna(teks): return np.nan
    teks = str(teks).strip()
    cocok = re.search(r'([\d]+\.?[\d]*)', teks)
    return float(cocok.group(1)) if cocok else np.nan

# ── Load & Preprocessing ──────────────────────────────────────
print("📂 Memuat dataset...")
df_raw = pd.read_excel(DATASET_PATH)

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

df['Usia'] = df['Usia'].apply(parse_usia)
for k in ['Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']:
    df[k] = df[k].apply(parse_nilai_lab)

df['Jenis_Kelamin'] = df['Jenis_Kelamin'].astype(str).str.strip().str.upper()
df['ICD'] = df['ICD'].astype(str).str.strip().str.upper()

kolom_numerik     = ['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']
kolom_kategorikal = ['Jenis_Kelamin']
df[kolom_numerik]     = SimpleImputer(strategy='median').fit_transform(df[kolom_numerik])
df[kolom_kategorikal] = SimpleImputer(strategy='most_frequent').fit_transform(df[kolom_kategorikal])

df = df.drop_duplicates(subset=kolom_numerik + kolom_kategorikal + ['ICD']).reset_index(drop=True)
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].replace({
    'LAKI-LAKI': 'L','LAKI LAKI':'L','PRIA':'L','M':'L',
    'PEREMPUAN': 'P','WANITA':'P','F':'P'
})
df = df[df['Jenis_Kelamin'].isin(['L','P'])].reset_index(drop=True)
df = df[df['ICD'].isin(['A90','A91'])].copy()
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].map({'L': 1, 'P': 0})
df['ICD'] = df['ICD'].map({'A91': 1, 'A90': 0})

X = df[['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit', 'Jenis_Kelamin']]
y = df['ICD'].astype(int)
print(f"✅ Data bersih: {len(df)} baris | A90={( y==0).sum()} | A91={(y==1).sum()}")

# ── Train / Test Split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ── GridSearch SMOTE ──────────────────────────────────────────
print("🔍 GridSearchCV SMOTE...")
param_grid_smote = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
gs_smote = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid_smote, scoring='recall_macro', cv=cv, n_jobs=-1
)
gs_smote.fit(X_train_smote, y_train_smote)
model_smote = gs_smote.best_estimator_
recall_smote = recall_score(y_test, model_smote.predict(X_test), average='macro')

# ── GridSearch Class Weight ───────────────────────────────────
print("🔍 GridSearchCV Class Weight...")
param_grid_cw = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'class_weight': ['balanced', None]
}
gs_cw = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid_cw, scoring='recall_macro', cv=cv, n_jobs=-1
)
gs_cw.fit(X_train, y_train)
model_cw = gs_cw.best_estimator_
recall_cw = recall_score(y_test, model_cw.predict(X_test), average='macro')

# ── Pilih Model Terbaik ───────────────────────────────────────
if recall_cw > recall_smote:
    model_final = model_cw
    nama_model  = "Decision Tree + Class Weight"
    X_fit, y_fit = X_train, y_train
    print(f"🏆 Model terpilih: {nama_model} (Recall Macro={recall_cw:.4f})")
else:
    model_final = model_smote
    nama_model  = "Decision Tree + SMOTE"
    X_fit, y_fit = X_train_smote, y_train_smote
    print(f"🏆 Model terpilih: {nama_model} (Recall Macro={recall_smote:.4f})")

model_final.fit(X_fit, y_fit)

FEATURE_NAMES = ['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit', 'Jenis Kelamin']
CLASS_NAMES   = ['A90 (DD)', 'A91 (DBD)']
COLORS        = ['#3B82F6', '#EF4444']   # biru = A90, merah = A91

depth = model_final.get_depth()
n_leaves = model_final.get_n_leaves()
print(f"🌲 Kedalaman pohon: {depth} | Jumlah daun: {n_leaves}")

# ── Visualisasi ───────────────────────────────────────────────
fig_w = max(30, n_leaves * 2.5)
fig_h = max(20, depth * 3.5)
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
fig.patch.set_facecolor('#0F172A')
ax.set_facecolor('#0F172A')

plot_tree(
    model_final,
    feature_names=FEATURE_NAMES,
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    ax=ax,
    fontsize=8,
    precision=3,
)

# Pewarnaan node
for artist in ax.get_children():
    if hasattr(artist, 'get_facecolor'):
        fc = artist.get_facecolor()
        if fc is not None:
            artist.set_edgecolor('#334155')
            artist.set_linewidth(0.8)

# Judul
ax.set_title(
    f"Pohon Keputusan (Decision Tree) — Klasifikasi Kode ICD DBD\n"
    f"Model: {nama_model}  |  Kedalaman: {depth}  |  Daun: {n_leaves}  |  "
    f"Data: {len(df)} sampel",
    fontsize=14, fontweight='bold',
    color='white', pad=20
)

# Legend
patch_a90 = mpatches.Patch(color=COLORS[0], label='A90 — Demam Dengue (DD)')
patch_a91 = mpatches.Patch(color=COLORS[1], label='A91 — Demam Berdarah Dengue (DBD)')
ax.legend(
    handles=[patch_a90, patch_a91],
    loc='upper right',
    fontsize=11,
    framealpha=0.85,
    facecolor='#1E293B',
    edgecolor='#475569',
    labelcolor='white'
)

plt.tight_layout(pad=2.0)
plt.savefig(OUTPUT_PATH, bbox_inches='tight', facecolor='#0F172A', dpi=150)
print(f"\n✅ Gambar Decision Tree disimpan: {OUTPUT_PATH}")
plt.close()
