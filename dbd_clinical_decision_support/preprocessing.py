"""
preprocessing.py
================
Modul Preprocessing untuk CDSS DBD

Bertanggung jawab atas:
- Validasi input pengguna
- Pembentukan DataFrame input untuk inference Pipeline model (SimpleImputer -> Decision Tree)
"""

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "Usia",
    "Trombosit",
    "Hematokrit",
    "Hemoglobin",
    "Leukosit",
    "Jenis_Kelamin"
]

CLINICAL_RANGES = {
    "Trombosit": (150000, 450000),    # /μL (nilai penuh)
    "Hematokrit": (33, 52),           # %
    "Hemoglobin": (10.0, 18.0),       # g/dL
    "Leukosit": (4000, 11000),        # /μL (nilai penuh)
    "Usia": (0, 120),                 # tahun
}


def validate_input(
    usia: float,
    jenis_kelamin: str,
    trombosit: float,
    hematokrit: float,
    hemoglobin: float,
    leukosit: float
) -> tuple[bool, list[str]]:
    
    errors = []
    fields = {
        "Usia": usia,
        "Trombosit": trombosit,
        "Hematokrit": hematokrit,
        "Hemoglobin": hemoglobin,
        "Leukosit": leukosit,
    }
    for name, val in fields.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            errors.append(f"❌ {name} harus diisi.")

    if not jenis_kelamin or jenis_kelamin not in ["Laki-laki", "Perempuan"]:
        errors.append("❌ Jenis Kelamin harus dipilih.")

    if usia is not None and not np.isnan(usia) and usia <= 0:
        errors.append("❌ Usia harus bernilai positif (> 0 tahun).")
    if trombosit is not None and not np.isnan(trombosit) and trombosit <= 0:
        errors.append("❌ Trombosit harus bernilai positif.")
    if hematokrit is not None and not np.isnan(hematokrit) and hematokrit <= 0:
        errors.append("❌ Hematokrit harus bernilai positif.")
    if hemoglobin is not None and not np.isnan(hemoglobin) and hemoglobin <= 0:
        errors.append("❌ Hemoglobin harus bernilai positif.")
    if leukosit is not None and not np.isnan(leukosit) and leukosit <= 0:
        errors.append("❌ Leukosit harus bernilai positif.")

    return len(errors) == 0, errors

def encode_jenis_kelamin(jenis_kelamin: str) -> int:
    return 1 if jenis_kelamin == "Laki-laki" else 0

def preprocess_input(
    usia: float,
    jenis_kelamin: str,
    trombosit: float,
    hematokrit: float,
    hemoglobin: float,
    leukosit: float
) -> tuple[bool, list[str], pd.DataFrame]:
    
    is_valid, errors = validate_input(
        usia, jenis_kelamin, trombosit, hematokrit, hemoglobin, leukosit
    )

    if not is_valid:
        return False, errors, None

    jk_encoded = encode_jenis_kelamin(jenis_kelamin)
    
    # Model pipeline requires a pandas DataFrame with specific column names
    df = pd.DataFrame([{
        "Usia": float(usia),
        "Trombosit": float(trombosit),
        "Hematokrit": float(hematokrit),
        "Hemoglobin": float(hemoglobin),
        "Leukosit": float(leukosit),
        "Jenis_Kelamin": float(jk_encoded)
    }])
    
    return True, [], df


ICD_LABELS = {
    0: {
        "kode": "A90",
        "nama": "Dengue Fever",
        "nama_id": "Demam Dengue",
        "risiko": "Risiko Klinis Lebih Rendah",
        "deskripsi": "Dengue Fever adalah infeksi virus dengue tanpa manifestasi perdarahan atau kebocoran plasma yang signifikan. Pasien umumnya menunjukkan demam tinggi mendadak, nyeri kepala, nyeri retro-orbital, dan nyeri otot/sendi.",
        "tindakan": "Pemantauan tanda vital secara berkala. Rehidrasi oral atau intravena sesuai kondisi. Pemberian antipiretik. Pemantauan jumlah trombosit dan hematokrit setiap hari.",
    },
    1: {
        "kode": "A91",
        "nama": "Dengue Hemorrhagic Fever",
        "nama_id": "Demam Berdarah Dengue",
        "risiko": "Risiko Klinis Lebih Tinggi",
        "deskripsi": "Dengue Hemorrhagic Fever (DHF) adalah bentuk yang lebih berat dari infeksi dengue, ditandai dengan trombositopenia (<100.000/μL), manifestasi perdarahan, dan kebocoran plasma yang dibuktikan dengan peningkatan hematokrit ≥20%.",
        "tindakan": "Rawat inap segera. Pemantauan ketat tanda-tanda syok dan perdarahan. Resusitasi cairan intravena. Transfusi trombosit jika diperlukan. Konsultasi dokter spesialis penyakit dalam atau anak.",
    },
}

def get_clinical_interpretation(predicted_class: int) -> dict:
    return ICD_LABELS.get(predicted_class, ICD_LABELS[0])
