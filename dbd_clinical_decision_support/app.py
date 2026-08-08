"""
app.py
======
Entry Point - Dashboard Utama CDSS DBD
Sistem Pendukung Keputusan Klinis: Klasifikasi ICD-10 A90 / A91

Tahap CRISP-DM: Deployment
Algoritma    : Decision Tree
Framework    : Streamlit

Jalankan:
    streamlit run app.py
"""

import os
import sys
import streamlit as st

# ─────────────────────────────────────────────
#  Konfigurasi Halaman (HARUS di baris pertama)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="SPK DBD | ICD-10 A90 & A91",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': (
            "**Sistem Pendukung Keputusan Klinis DBD**\n\n"
            "Klasifikasi ICD-10 A90 (Dengue Fever) dan A91 (Dengue Hemorrhagic Fever)\n\n"
            "Menggunakan algoritma Decision Tree dengan metodologi CRISP-DM.\n\n"
            "⚠️ Sistem ini hanya sebagai pendukung keputusan klinis, "
            "bukan pengganti diagnosis dokter."
        )
    }
)

# ─────────────────────────────────────────────
#  Load Custom CSS
# ─────────────────────────────────────────────

def load_css():
    """Load file CSS custom dari folder assets."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────────
#  Import Modul
# ─────────────────────────────────────────────

import joblib
import numpy as np
from streamlit_option_menu import option_menu

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "pipeline_dbd.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "eval_metrics.pkl")


# ─────────────────────────────────────────────
#  Helper: Load Model
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load pipeline model dari file .pkl.
    Cache resource agar tidak reload setiap interaksi.
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Fallback: cek nama lain
    fallback = os.path.join(BASE_DIR, "models", "decision_tree.pkl")
    if os.path.exists(fallback):
        return joblib.load(fallback)
    return None


@st.cache_resource(show_spinner=False)
def load_metrics():
    """Load metrik evaluasi dari file .pkl."""
    if os.path.exists(METRICS_PATH):
        return joblib.load(METRICS_PATH)
    return None


# ─────────────────────────────────────────────
#  Sidebar Navigation
# ─────────────────────────────────────────────

with st.sidebar:
    # Logo & Branding
    st.markdown("""
    <div class='sidebar-logo'>
        <div class='sidebar-logo-icon'>🏥</div>
        <div class='sidebar-logo-title'>SPK DBD</div>
        <div class='sidebar-logo-sub'>ICD-10 A90 &amp; A91 Classifier</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-nav-label'>Navigation</div>", unsafe_allow_html=True)

    selected_page = option_menu(
        menu_title=None,
        options=["Dashboard", "Prediksi Pasien", "Evaluasi Model", "Tentang Sistem"],
        icons=["house-fill", "clipboard2-pulse-fill", "graph-up-arrow", "info-circle-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0 8px", "background-color": "#1D4ED8"},
            "icon": {"color": "rgba(255,255,255,0.75)", "font-size": "14px"},
            "nav-link": {
                "font-size": "0.875rem",
                "text-align": "left",
                "margin": "2px 0",
                "color": "rgba(255,255,255,0.82)",
                "border-radius": "8px",
                "padding": "10px 14px",
                "font-weight": "500",
            },
            "nav-link-selected": {
                "background-color": "rgba(255,255,255,0.18)",
                "color": "white",
                "font-weight": "700",
                "border-left": "3px solid rgba(255,255,255,0.8)",
            },
        }
    )

    # Status model di sidebar
    model = load_model()
    metrics = load_metrics()

    st.markdown("""
    <div style='margin: 0.75rem 0 0.25rem;border-top:1px solid rgba(255,255,255,0.1);'></div>
    <div class='sidebar-nav-label'>Informasi Sistem</div>
    """, unsafe_allow_html=True)

    if model is not None:
        st.markdown("""
        <div class='sidebar-info'>
            <div class='sidebar-info-label'>Status Model</div>
            <div class='sidebar-info-row'>
                <span class='sidebar-info-key'>Model</span>
                <span class='sidebar-info-val'>✅ Aktif</span>
            </div>
            <div class='sidebar-info-row'>
                <span class='sidebar-info-key'>Dataset</span>
                <span class='sidebar-info-val'>RS Aulia</span>
            </div>
            <div class='sidebar-info-row'>
                <span class='sidebar-info-key'>Algoritma</span>
                <span class='sidebar-info-val'>Decision Tree</span>
            </div>
            <div class='sidebar-info-row'>
                <span class='sidebar-info-key'>Metodologi</span>
                <span class='sidebar-info-val'>CRISP-DM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='sidebar-info'>
            <div style='font-size:0.82rem; color:#FCD34D; font-weight:600;'>⚠️ Model Belum Tersedia</div>
            <div style='font-size:0.72rem; color:rgba(255,255,255,0.6); margin-top:0.3rem;'>
                Jalankan: python train_model.py
            </div>
        </div>
        """, unsafe_allow_html=True)




# ═══════════════════════════════════════════════════════════
#  ROUTING HALAMAN
# ═══════════════════════════════════════════════════════════

if selected_page == "Dashboard":
    # ─── HALAMAN DASHBOARD ───────────────────────────────

    # ── Hero Section ──
    st.markdown("""
    <div class='hero-card'>
        <div class='hero-badge'>🏥 SPK DBD</div>
        <h1>Sistem Pendukung Keputusan Klinis DBD</h1>
        <p>
            Klasifikasi pasien Demam Berdarah Dengue berdasarkan kode ICD-10 <strong>A90</strong>
            dan <strong>A91</strong> menggunakan algoritma Decision Tree dengan metodologi CRISP-DM.
        </p>
        <div class='hero-tags'>
            <span class='hero-tag'>Decision Tree</span>
            <span class='hero-tag'>CRISP-DM</span>
            <span class='hero-tag'>ICD-10 Classifier</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Warning CDSS disclaimer ──
    st.markdown("""
    <div class='warning-banner'>
        <span style='font-size:1.1rem;'>⚠️</span>
        <p><strong>Perhatian:</strong> Sistem ini berfungsi sebagai <em>Clinical Decision Support System (CDSS)</em>
        dan <strong>bukan</strong> sebagai pengganti diagnosis dokter. Keputusan klinis akhir tetap berada
        di tangan tenaga medis yang berwenang.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──
    if metrics:
        st.markdown("<div class='section-header'><h2>📈 Ringkasan Performa Model</h2></div>",
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            acc = metrics.get('accuracy', 0)
            st.markdown(f"""
            <div class='metric-card blue'>
                <div class='metric-card-icon'>🎯</div>
                <div class='metric-card-value'>{acc*100:.1f}%</div>
                <div class='metric-card-label'>Accuracy</div>
                <div class='metric-card-sub'>Overall accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            prec = metrics.get('precision', 0)
            st.markdown(f"""
            <div class='metric-card green'>
                <div class='metric-card-icon'>🔍</div>
                <div class='metric-card-value'>{prec*100:.1f}%</div>
                <div class='metric-card-label'>Precision</div>
                <div class='metric-card-sub'>Macro average</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            rec = metrics.get('recall', 0)
            st.markdown(f"""
            <div class='metric-card amber'>
                <div class='metric-card-icon'>📡</div>
                <div class='metric-card-value'>{rec*100:.1f}%</div>
                <div class='metric-card-label'>Recall</div>
                <div class='metric-card-sub'>Macro average</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            f1 = metrics.get('f1_score', 0)
            st.markdown(f"""
            <div class='metric-card red'>
                <div class='metric-card-icon'>⚖️</div>
                <div class='metric-card-value'>{f1*100:.1f}%</div>
                <div class='metric-card-label'>F1-Score</div>
                <div class='metric-card-sub'>Macro average</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ Model belum dilatih. Jalankan `python train_model.py` untuk melatih model.")

    # ── Quick Action Cards ──
    st.markdown("<div class='section-header'><h2>🚀 Menu Utama</h2></div>",
                unsafe_allow_html=True)

    qa1, qa2, qa3 = st.columns(3)

    with qa1:
        st.markdown("""
        <div class='action-card'>
            <span class='action-card-icon'>🩺</span>
            <div class='action-card-title'>Prediksi Pasien</div>
            <p class='action-card-desc'>
                Masukkan data laboratorium pasien untuk mendapatkan klasifikasi ICD-10
                secara otomatis beserta confidence score dan interpretasi klinis.
            </p>
            <div class='action-card-arrow'>Mulai Prediksi →</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶  Mulai Prediksi", type="primary", use_container_width=True, key="btn_prediksi_dashboard"):
            st.switch_page("pages/1_Prediksi.py")

    with qa2:
        st.markdown("""
        <div class='action-card'>
            <span class='action-card-icon'>📊</span>
            <div class='action-card-title'>Evaluasi Model</div>
            <p class='action-card-desc'>
                Lihat performa model secara detail: metrik evaluasi, confusion matrix,
                classification report, dan feature importance.
            </p>
            <div class='action-card-arrow'>Lihat Evaluasi →</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📈 Lihat Evaluasi", use_container_width=True, key="btn_eval_dashboard"):
            st.switch_page("pages/2_Evaluasi_Model.py")

    with qa3:
        st.markdown("""
        <div class='action-card'>
            <span class='action-card-icon'>ℹ️</span>
            <div class='action-card-title'>Tentang Sistem</div>
            <p class='action-card-desc'>
                Pelajari metodologi CRISP-DM, cara kerja algoritma Decision Tree,
                dan arsitektur sistem secara menyeluruh.
            </p>
            <div class='action-card-arrow'>Pelajari Sistem →</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📖 Pelajari Sistem", use_container_width=True, key="btn_about_dashboard"):
            st.switch_page("pages/3_Tentang_Sistem.py")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Penjelasan Sistem ──
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("<div class='section-header'><h2>🔬 Tentang Penelitian</h2></div>",
                    unsafe_allow_html=True)

        with st.expander("📌 Latar Belakang & Tujuan", expanded=True):
            st.markdown("""
            Demam Berdarah Dengue (DBD) merupakan penyakit endemik yang menjadi tantangan 
            besar bagi tenaga kesehatan, khususnya dalam **membedakan Dengue Fever (A90)** 
            dengan **Dengue Hemorrhagic Fever (A91)** secara cepat dan akurat.

            Sistem ini dibangun sebagai **Clinical Decision Support System (CDSS)** yang 
            membantu tenaga medis mengklasifikasikan pasien DBD menggunakan data laboratorium 
            sederhana melalui algoritma **Decision Tree**.
            """)

        with st.expander("🔬 Fitur Input & Target Prediksi"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("""
                **6 Variabel Input:**
                | No | Variabel | Satuan |
                |----|----------|--------|
                | 1 | Usia | Tahun |
                | 2 | Jenis Kelamin | L/P |
                | 3 | Trombosit | ribu/μL |
                | 4 | Hematokrit | % |
                | 5 | Hemoglobin | g/dL |
                | 6 | Leukosit | ribu/μL |
                """)
            with col_b:
                st.markdown("""
                **Target Klasifikasi:**
                | Label | Kode ICD | Diagnosis |
                |-------|----------|-----------|
                | 0 | A90 | Dengue Fever |
                | 1 | A91 | Dengue Hemorrhagic Fever |
                """)

        with st.expander("🧠 Algoritma Decision Tree"):
            st.markdown("""
            **Decision Tree** adalah algoritma supervised learning berbasis pohon keputusan 
            yang membagi dataset berdasarkan kondisi pada setiap fitur secara rekursif.

            **Keunggulan dalam konteks klinis:**
            - 🔍 **Interpretable** — Tenaga medis dapat memahami alur keputusan
            - ⚡ **Efisien** — Prediksi real-time dari data laboratorium
            - 📊 **Feature Importance** — Menunjukkan variabel mana yang paling berpengaruh
            - ⚖️ **Class Weight** — Menangani ketidakseimbangan kelas A90 vs A91

            **Hyperparameter (GridSearchCV):**
            - `criterion`: gini / entropy
            - `max_depth`: 3, 5, 7, 10, None
            - `min_samples_split`: 2, 5, 10
            - `class_weight`: balanced
            """)

    with col2:
        st.markdown("<div class='section-header'><h2>🔄 Alur CRISP-DM</h2></div>",
                    unsafe_allow_html=True)

        crisp_phases = [
            ("01", "Business Understanding",
             "Mendefinisikan tujuan klasifikasi DBD menjadi ICD-10 A90/A91 untuk mendukung keputusan klinis."),
            ("02", "Data Understanding",
             "Eksplorasi dataset 3.000 pasien RS Aulia: distribusi kelas, pola data laboratorium, missing values."),
            ("03", "Data Preparation",
             "Parsing format data, encoding Jenis Kelamin, penanganan missing value, normalisasi fitur."),
            ("04", "Modeling",
             "Training Decision Tree dengan Pipeline StandardScaler + DecisionTreeClassifier + GridSearchCV."),
            ("05", "Evaluation",
             "Evaluasi metrik Accuracy, Precision, Recall, F1-Score. Prioritas: Recall kelas A91."),
            ("06", "Deployment",
             "Integrasi model ke aplikasi Streamlit sebagai CDSS untuk tenaga medis RS Aulia."),
        ]

        for num, title, desc in crisp_phases:
            st.markdown(f"""
            <div class='phase-card'>
                <div class='phase-num'>{num}</div>
                <div class='phase-content'>
                    <div class='phase-title'>{title}</div>
                    <p class='phase-desc'>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class='footer-text'>
        Sistem Pendukung Keputusan Klinis DBD &bull; Implementasi CRISP-DM &bull;
        Dataset RS Aulia &bull; Algoritma Decision Tree
    </div>
    """, unsafe_allow_html=True)



elif selected_page == "Prediksi Pasien":
    st.switch_page("pages/1_Prediksi.py")

elif selected_page == "Evaluasi Model":
    st.switch_page("pages/2_Evaluasi_Model.py")

elif selected_page == "Tentang Sistem":
    st.switch_page("pages/3_Tentang_Sistem.py")
