import sys
sys.stdout.reconfigure(encoding='utf-8')
import re, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer

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

df_raw = pd.read_excel('Data_Lab_Penyakit_DBD_RS_Aulia.xlsx')
print(f"DATA RAW  : {len(df_raw)} baris, {df_raw.shape[1]} kolom")
print(f"Kolom     : {list(df_raw.columns)}")
print()

kolom_dipilih = {
    'Usia (tahun)': 'Usia',
    'Jenis Kelamin\n(L/P)': 'Jenis_Kelamin',
    'Trombosit\n(ribu/\u03bcL)': 'Trombosit',
    'Hematokrit (%)': 'Hematokrit',
    'Hemoglobin (g/dL)': 'Hemoglobin',
    'Leukosit\n(ribu/\u03bcL)': 'Leukosit',
    'kode ICD': 'ICD'
}

available = {k: v for k, v in kolom_dipilih.items() if k in df_raw.columns}
print(f"Kolom yang ditemukan: {list(available.keys())}")

df = df_raw[list(available.keys())].rename(columns=available).copy()

if 'Usia' in df.columns:
    df['Usia'] = df['Usia'].apply(parse_usia)
for kolom in ['Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit']:
    if kolom in df.columns:
        df[kolom] = df[kolom].apply(parse_nilai_lab)

df['Jenis_Kelamin'] = df['Jenis_Kelamin'].astype(str).str.strip().str.upper()
df['ICD'] = df['ICD'].astype(str).str.strip().str.upper()

kolom_lab = [c for c in ['Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit'] if c in df.columns]
df['_tanpa_data_lab'] = df[kolom_lab].isnull().all(axis=1)
tanpa_lab = df['_tanpa_data_lab'].sum()
print(f"Baris tanpa data lab sama sekali: {tanpa_lab}")

print("\nMissing values sebelum imputasi:")
mv = df.isnull().sum()
print(mv[mv > 0])

kolom_numerik = [c for c in ['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit'] if c in df.columns]
kolom_kategorikal = ['Jenis_Kelamin']
df[kolom_numerik] = SimpleImputer(strategy='median').fit_transform(df[kolom_numerik])
df[kolom_kategorikal] = SimpleImputer(strategy='most_frequent').fit_transform(df[kolom_kategorikal])

sebelum_dedup = len(df)
df = df.drop_duplicates(subset=kolom_numerik + ['Jenis_Kelamin', 'ICD']).reset_index(drop=True)
print(f"\nSetelah imputasi: {sebelum_dedup} baris")
print(f"Setelah hapus duplikat: {len(df)} baris (duplikat dihapus: {sebelum_dedup - len(df)})")

df['Jenis_Kelamin'] = df['Jenis_Kelamin'].replace({
    'LAKI-LAKI': 'L', 'LAKI LAKI': 'L', 'PRIA': 'L', 'M': 'L',
    'PEREMPUAN': 'P', 'WANITA': 'P', 'F': 'P'
})
df = df[df['Jenis_Kelamin'].isin(['L', 'P'])].reset_index(drop=True)
print(f"Setelah filter JK valid: {len(df)} baris")

print(f"\nDistribusi ICD (semua nilai):")
print(df['ICD'].value_counts())

df_valid = df[df['ICD'].isin(['A90', 'A91'])].copy()
n_a90 = (df_valid['ICD'] == 'A90').sum()
n_a91 = (df_valid['ICD'] == 'A91').sum()

print(f"\n{'='*50}")
print(f"DATA BERSIH SIAP MODEL: {len(df_valid)} baris")
print(f"  - A90 (Demam Dengue / DD)    : {n_a90} baris")
print(f"  - A91 (Demam Berdarah / DBD) : {n_a91} baris")
print(f"{'='*50}")
print(f"\nSplit 80:20:")
print(f"  - Train set : ~{int(len(df_valid)*0.8)} baris")
print(f"  - Test set  : ~{int(len(df_valid)*0.2)} baris")
