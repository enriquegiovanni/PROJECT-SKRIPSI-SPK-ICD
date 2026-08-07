"""
pages/1_Prediksi.py
===================
Halaman Prediksi Pasien - CDSS DBD
Form input data laboratorium → Prediksi ICD-10 A90/A91

Tahap CRISP-DM: Deployment
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ── Path Setup ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from preprocessing import (
    preprocess_input, get_clinical_interpretation,
    FEATURE_NAMES, CLINICAL_RANGES
)

# ── Page Config ──
st.set_page_config(
    page_title="Prediksi Pasien | CDSS DBD",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Load CSS ──
def load_css():
    css_path = os.path.join(BASE_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ── Load Model ──
MODEL_PATH  = os.path.join(BASE_DIR, "models", "pipeline_dbd.pkl")
FALLBACK    = os.path.join(BASE_DIR, "models", "decision_tree.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    if os.path.exists(FALLBACK):
        return joblib.load(FALLBACK)
    return None


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
        <div class='sidebar-logo-icon'>🏥</div>
        <div class='sidebar-logo-title'>SPK DBD</div>
        <div class='sidebar-logo-sub'>ICD-10 A90 &amp; A91 Classifier</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-nav-label'>Navigation</div>", unsafe_allow_html=True)

    from streamlit_option_menu import option_menu
    selected_page = option_menu(
        menu_title=None,
        options=["Dashboard", "Prediksi Pasien", "Evaluasi Model", "Tentang Sistem"],
        icons=["house-fill", "clipboard2-pulse-fill", "graph-up-arrow", "info-circle-fill"],
        default_index=1,
        styles={
            "container": {"padding": "0 8px", "background-color": "#1D4ED8"},
            "icon": {"color": "rgba(255,255,255,0.75)", "font-size": "14px"},
            "nav-link": {
                "font-size": "0.875rem", "text-align": "left", "margin": "2px 0",
                "color": "rgba(255,255,255,0.82)", "border-radius": "8px", "padding": "10px 14px",
                "font-weight": "500",
            },
            "nav-link-selected": {
                "background-color": "rgba(255,255,255,0.18)", "color": "white", "font-weight": "700",
                "border-left": "3px solid rgba(255,255,255,0.8)",
            },
        }
    )
    if selected_page == "Dashboard":
        st.switch_page("app.py")
    elif selected_page == "Evaluasi Model":
        st.switch_page("pages/2_Evaluasi_Model.py")
    elif selected_page == "Tentang Sistem":
        st.switch_page("pages/3_Tentang_Sistem.py")

    # Referensi nilai normal
    st.markdown("""
    <div style='margin: 0.75rem 0 0.25rem;border-top:1px solid rgba(255,255,255,0.1);'></div>
    <div class='sidebar-nav-label'>Nilai Normal Lab</div>
    """, unsafe_allow_html=True)

    normal_refs = [
        ("Trombosit", "150.000–450.000 /μL"),
        ("Hematokrit", "35–52 %"),
        ("Hemoglobin", "10–18 g/dL"),
        ("Leukosit", "4.000–11.000 /μL"),
    ]

    for name, ref in normal_refs:
        st.markdown(f"""
        <div class='sidebar-info-row' style='padding:0 1rem 0.3rem;'>
            <span class='sidebar-info-key'>{name}</span>
            <span class='sidebar-info-val'>{ref}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-footer'>
        <p>CDSS DBD v1.0 &bull; RS Aulia<br>Metodologi CRISP-DM</p>
    </div>
    """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════

model = load_model()

# ── Cek Model ──
if model is None:
    st.markdown("""
    <div class='hero-card'>
        <div class='hero-badge'>⚠️ Peringatan</div>
        <h1>Model Belum Tersedia</h1>
        <p>Model belum tersedia. Silakan lakukan proses training terlebih dahulu.</p>
    </div>
    """, unsafe_allow_html=True)
    st.code("python train_model.py", language="bash")
    st.stop()

# ── Initialize Session State ──
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "input_data" not in st.session_state:
    st.session_state.input_data = {}


# ─────────────────────────────────────────────
#  TAMPILAN: FORM INPUT
# ─────────────────────────────────────────────

if not st.session_state.prediction_done:

    # ── Hero Header ──
    st.markdown("""
    <div class='hero-card'>
        <div class='hero-badge'>🩺 Prediksi ICD-10</div>
        <h1>Prediksi Klasifikasi Pasien DBD</h1>
        <p>
            Masukkan data demografis dan hasil laboratorium pasien untuk mendapatkan prediksi
            klasifikasi ICD-10 A90 (Dengue Fever) atau A91 (Dengue Hemorrhagic Fever)
            secara otomatis.
        </p>
        <div class='hero-tags'>
            <span class='hero-tag'>Data Laboratorium</span>
            <span class='hero-tag'>Decision Tree</span>
            <span class='hero-tag'>ICD-10 A90/A91</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Warning disclaimer ──
    st.markdown("""
    <div class='warning-banner'>
        <span style='font-size:1.1rem;'>⚠️</span>
        <p><strong>Perhatian CDSS:</strong> Hasil prediksi hanya bersifat sebagai
        <em>pendukung keputusan klinis</em>. Diagnosis akhir tetap menjadi tanggung jawab
        dokter yang menangani pasien.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Form Input ──
    st.markdown("""
    <div class='section-title'>📋 Data Laboratorium Pasien</div>
    <div class='section-desc'>Lengkapi semua field bertanda * di bawah ini dengan benar.</div>
    """, unsafe_allow_html=True)

    with st.form("form_prediksi", clear_on_submit=False):

        # ── Seksi 1: Data Demografis ──
        st.markdown("""
        <div style='background:#F0F4FF; border-radius:8px; padding:0.6rem 1rem;
                    border-left:4px solid #2563EB; margin-bottom:1rem;'>
            <span style='font-size:0.85rem; font-weight:700; color:#2563EB;
                         text-transform:uppercase; letter-spacing:0.06em;'>
                👤 Data Demografis
            </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            usia = st.number_input(
                label="Usia (Tahun) *",
                min_value=0,
                max_value=120,
                value=None,
                step=1,
                help="Usia pasien dalam satuan tahun",
                key="input_usia"
            )
            st.caption("📌 Contoh: 25 tahun")

        with col2:
            jenis_kelamin = st.selectbox(
                label="Jenis Kelamin *",
                options=["", "Laki-laki", "Perempuan"],
                index=0,
                help="Pilih jenis kelamin pasien",
                key="input_jk"
            )


        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # ── Seksi 2: Hasil Laboratorium ──
        st.markdown("""
        <div style='background:#F0FFF4; border-radius:8px; padding:0.6rem 1rem;
                    border-left:4px solid #16A34A; margin-bottom:1rem;'>
            <span style='font-size:0.85rem; font-weight:700; color:#16A34A;
                         text-transform:uppercase; letter-spacing:0.06em;'>
                🧪 Hasil Pemeriksaan Laboratorium
            </span>
        </div>
        """, unsafe_allow_html=True)

        col3, col4, col5, col6 = st.columns(4)

        with col3:
            trombosit = st.number_input(
                label="Trombosit (/μL) *",
                min_value=0,
                max_value=500000,
                value=None,
                step=1000,
                format="%d",
                help="Nilai trombosit dalam satuan /μL (contoh: 137000)",
                key="input_trombosit"
            )
            st.caption("ℹ️ Normal: 150.000–450.000\nContoh: 137.000")

        with col4:
            hematokrit = st.number_input(
                label="Hematokrit (%) *",
                min_value=0.0,
                max_value=80.0,
                value=None,
                step=0.1,
                format="%.1f",
                help="Nilai hematokrit dalam persen",
                key="input_hematokrit"
            )
            st.caption("ℹ️ Normal: 35–52%\nContoh: 38.0")

        with col5:
            hemoglobin = st.number_input(
                label="Hemoglobin (g/dL) *",
                min_value=0.0,
                max_value=25.0,
                value=None,
                step=0.1,
                format="%.1f",
                help="Nilai hemoglobin dalam g/dL",
                key="input_hemoglobin"
            )
            st.caption("ℹ️ Normal: 10–18 g/dL\nContoh: 12.8")

        with col6:
            leukosit = st.number_input(
                label="Leukosit (/μL) *",
                min_value=0,
                max_value=100000,
                value=None,
                step=100,
                format="%d",
                help="Jumlah leukosit dalam satuan /μL (contoh: 5400)",
                key="input_leukosit"
            )
            st.caption("ℹ️ Normal: 4.000–11.000\nContoh: 5.400")


        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # ── Submit Button ──
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submitted = st.form_submit_button(
                "🔬 Prediksi Klasifikasi ICD-10",
                type="primary",
                use_container_width=True
            )

    # ── Proses Prediksi ──
    if submitted:
        # Konversi nilai None ke float NaN untuk validasi
        usia_val       = float(usia) if usia is not None else float('nan')
        trombosit_val  = float(trombosit) if trombosit is not None else float('nan')
        hematokrit_val = float(hematokrit) if hematokrit is not None else float('nan')
        hemoglobin_val = float(hemoglobin) if hemoglobin is not None else float('nan')
        leukosit_val   = float(leukosit) if leukosit is not None else float('nan')

        # Validasi
        is_valid, errors, features = preprocess_input(
            usia_val, jenis_kelamin, trombosit_val,
            hematokrit_val, hemoglobin_val, leukosit_val
        )

        if not is_valid:
            st.error("**Terdapat kesalahan pada input:**")
            for err in errors:
                st.warning(err)
        else:
            # Jalankan prediksi
            with st.spinner("⚙️ Memproses prediksi..."):
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                confidence = float(probabilities[int(prediction)])

                # Feature importance
                try:
                    dt_step = model.named_steps.get('classifier', model)
                    feature_importances = dt_step.feature_importances_
                except Exception:
                    feature_importances = None

            # Simpan ke session state
            st.session_state.prediction_result = {
                "prediction"          : int(prediction),
                "probabilities"       : probabilities.tolist(),
                "confidence"          : confidence,
                "feature_importances" : feature_importances.tolist() if feature_importances is not None else None,
                "input"               : {
                    "Usia"           : usia_val,
                    "Jenis Kelamin"  : jenis_kelamin,
                    "Trombosit"      : trombosit_val,
                    "Hematokrit"     : hematokrit_val,
                    "Hemoglobin"     : hemoglobin_val,
                    "Leukosit"       : leukosit_val,
                }
            }
            st.session_state.prediction_done = True
            st.rerun()


# ─────────────────────────────────────────────
#  TAMPILAN: HASIL PREDIKSI
# ─────────────────────────────────────────────

else:
    result      = st.session_state.prediction_result
    pred_class  = result["prediction"]
    probs       = result["probabilities"]
    confidence  = result["confidence"]
    fi          = result.get("feature_importances")
    input_data  = result["input"]

    clinical    = get_clinical_interpretation(pred_class)
    icd_code    = clinical["kode"]
    icd_nama    = clinical["nama"]
    risiko      = clinical["risiko"]

    # ── Hero Header Hasil ──
    hero_class = "hero-card-danger" if pred_class == 1 else ""
    emoji      = "🔴" if pred_class == 1 else "🔵"

    st.markdown(f"""
    <div class='hero-card {hero_class}'>
        <div class='hero-badge'>{emoji} Hasil Prediksi</div>
        <h1>Prediksi Klasifikasi ICD-10 Selesai</h1>
        <p>Prediksi berhasil dilakukan berdasarkan data laboratorium yang dimasukkan.
        Lihat hasil lengkap di bawah ini.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── ICD Result + Metrics ──
    col_main, col_meta = st.columns([3, 2], gap="large")

    with col_main:
        # Kotak Hasil Utama
        card_class  = "result-card-a91" if pred_class == 1 else "result-card-a90"
        badge_class = "icd-badge-a91" if pred_class == 1 else "icd-badge-a90"
        title_color = "#DC2626" if pred_class == 1 else "#2563EB"

        st.markdown(f"""
        <div class='{card_class}'>
            <span class='icd-badge {badge_class}'>{icd_code}</span>
            <div class='icd-title' style='color:{title_color};'>{emoji} {icd_nama}</div>
            <div class='icd-risk'>⚠️ {risiko}</div>
            <hr style='border-color:rgba(0,0,0,0.08); margin:0.85rem 0;'>
            <p style='font-size:0.875rem; color:#374151; line-height:1.7; margin:0;'>
                {clinical["deskripsi"]}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Rekomendasi Tindakan Klinis
        st.markdown("#### 💊 Rekomendasi Tindakan Klinis")
        st.info(f"""
        **Berdasarkan klasifikasi {icd_code} — {icd_nama}:**

        {clinical["tindakan"]}

        *⚠️ Keputusan klinis akhir tetap berada pada tenaga medis yang menangani pasien.*
        """)

    with col_meta:
        # Confidence Score
        conf_pct = confidence * 100
        conf_color = "#DC2626" if pred_class == 1 else "#2563EB"

        st.markdown(f"""
        <div class='metric-card {"red" if pred_class == 1 else "blue"}' style='margin-bottom:1rem;'>
            <div class='metric-card-icon'>🎯</div>
            <div class='metric-card-value' style='color:{conf_color};'>{conf_pct:.1f}%</div>
            <div class='metric-card-label'>Confidence Score</div>
            <div class='metric-card-sub'>Keyakinan model terhadap prediksi</div>
        </div>
        """, unsafe_allow_html=True)

        # Probabilitas per kelas
        prob_a90 = probs[0] * 100
        prob_a91 = probs[1] * 100

        st.markdown(f"""
<div class='prob-container'>
<div class='prob-label'>Probabilitas Prediksi</div>
<div class='prob-row'>
<div class='prob-header'>
<span class='prob-name'>🔵 A90 — Dengue Fever</span>
<span class='prob-pct' style='color:#2563EB;'>{prob_a90:.1f}%</span>
</div>
<div class='prob-bar-bg' style='background:#DBEAFE;'>
<div class='prob-bar-fill' style='width:{prob_a90:.1f}%; background:linear-gradient(90deg,#1D4ED8,#3B82F6);'></div>
</div>
</div>
<div class='prob-row'>
<div class='prob-header'>
<span class='prob-name'>🔴 A91 — DHF</span>
<span class='prob-pct' style='color:#DC2626;'>{prob_a91:.1f}%</span>
</div>
<div class='prob-bar-bg' style='background:#FEE2E2;'>
<div class='prob-bar-fill' style='width:{prob_a91:.1f}%; background:linear-gradient(90deg,#991B1B,#EF4444);'></div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

        # Data Input Summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Ringkasan Input")
        input_df = pd.DataFrame([{
            "Variabel"   : k,
            "Nilai"      : f"{v}" if k == "Jenis Kelamin" else f"{v:.1f}",
        } for k, v in input_data.items()])
        st.dataframe(
            input_df.set_index("Variabel"),
            use_container_width=True,
            hide_index=False,
        )

    # ── Feature Importance ──
    if fi is not None:
        st.markdown("---")
        st.markdown("<div class='section-header'><h2>📊 Feature Importance</h2></div>",
                    unsafe_allow_html=True)
        st.caption("Variabel yang paling berpengaruh dalam keputusan klasifikasi model Decision Tree.")

        fi_arr   = np.array(fi)
        sorted_idx = np.argsort(fi_arr)[::-1]
        names_sorted = [FEATURE_NAMES[i] for i in sorted_idx]
        vals_sorted  = [fi_arr[i] for i in sorted_idx]

        col_fi1, col_fi2 = st.columns([2, 3], gap="large")

        with col_fi1:
            st.markdown("**Ranking Variabel:**")
            colors_rank = ["#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"]
            for rank, (name, val) in enumerate(zip(names_sorted, vals_sorted)):
                pct = val * 100
                color = colors_rank[min(rank, len(colors_rank)-1)]
                st.markdown(f"""
                <div style='margin-bottom:0.6rem;'>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:0.2rem;'>
                        <div style='display:flex; align-items:center; gap:0.4rem;'>
                            <span style='background:{color}; color:white; font-size:0.7rem;
                                         font-weight:700; width:22px; height:22px; border-radius:50%;
                                         display:inline-flex; align-items:center; justify-content:center;'>
                                {rank+1}
                            </span>
                            <span style='font-size:0.88rem; font-weight:600; color:#111827;'>{name}</span>
                        </div>
                        <span style='font-size:0.85rem; font-weight:700; color:{color};'>{pct:.1f}%</span>
                    </div>
                    <div style='background:#E5E7EB; border-radius:100px; height:8px; overflow:hidden;'>
                        <div style='width:{pct:.1f}%; height:100%; background:{color}; border-radius:100px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_fi2:
            # Bar Chart Plotly
            fig_fi = go.Figure(go.Bar(
                x=vals_sorted,
                y=names_sorted,
                orientation='h',
                marker=dict(
                    color=vals_sorted,
                    colorscale=[[0, '#BFDBFE'], [0.5, '#2563EB'], [1, '#1D4ED8']],
                    showscale=False,
                ),
                text=[f"{v:.3f}" for v in vals_sorted],
                textposition='outside',
                textfont=dict(size=12, color='#111827')
            ))
            fig_fi.update_layout(
                title=dict(
                    text="Feature Importance — Decision Tree",
                    font=dict(size=14, color='#111827', family='Inter')
                ),
                xaxis=dict(
                    title="Importance Score",
                    showgrid=True,
                    gridcolor='#F3F4F6',
                    tickfont=dict(size=11),
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=12, color='#111827'),
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=60, t=50, b=40),
                height=300,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── Tombol Prediksi Lagi ──
    st.markdown("---")
    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        if st.button("🔄 Prediksi Lagi", type="primary", use_container_width=True, key="btn_reset"):
            st.session_state.prediction_done = False
            st.session_state.prediction_result = None
            st.rerun()

    st.markdown("""
    <div class='footer-text'>
        ⚠️ Hasil prediksi ini hanya bersifat sebagai pendukung keputusan klinis.
        Diagnosis akhir merupakan wewenang dokter yang menangani.
    </div>
    """, unsafe_allow_html=True)
