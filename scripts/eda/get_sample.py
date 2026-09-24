from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "data" / "raw" / "Data_Lab_Penyakit_DBD_RS_Aulia.xlsx"

df = pd.read_excel(DATASET_PATH)
a91_df = df[df['kode ICD'] == 'A91']
if not a91_df.empty:
    sample = a91_df.iloc[0]
    print('Sample A91 Data:')
    print('Usia:', sample['Usia (tahun)'])
    print('Jenis Kelamin:', sample['Jenis Kelamin\n(L/P)'])
    print('Trombosit:', sample['Trombosit\n(ribu/μL)'])
    print('Hematokrit:', sample['Hematokrit (%)'])
    print('Hemoglobin:', sample['Hemoglobin (g/dL)'])
    print('Leukosit:', sample['Leukosit\n(ribu/μL)'])
