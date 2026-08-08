"""
pages/3_Tentang_Sistem.py
=========================
Halaman Tentang Sistem - CDSS DBD
Informasi tentang penelitian, metodologi CRISP-DM, dan arsitektur sistem.
"""

import os
import sys
import streamlit as st

# ── Path Setup ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ── Page Config ──
st.set_page_config(
    page_title="Tentang Sistem | CDSS DBD",
    page_icon="ℹ️",
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

# ── Load Model & Metrics ──
MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline_dbd.pkl")
FALLBACK = os.path.join(BASE_DIR, "models", "decision_tree.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "eval_metrics.pkl")

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    if os.path.exists(FALLBACK):
        return joblib.load(FALLBACK)
    return None

@st.cache_resource(show_spinner=False)
def load_metrics():
    if os.path.exists(METRICS_PATH):
        return joblib.load(METRICS_PATH)
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
        default_index=3,
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
    elif selected_page == "Prediksi Pasien":
        st.switch_page("pages/1_Prediksi.py")
    elif selected_page == "Evaluasi Model":
        st.switch_page("pages/2_Evaluasi_Model.py")

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
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════

# ── Hero ──
st.markdown("""
<div class='hero-card hero-card-teal'>
    <div class='hero-badge'>ℹ️ Tentang Sistem</div>
    <h1>Metodologi &amp; Arsitektur Sistem</h1>
    <p>
        Informasi detail mengenai arsitektur sistem, algoritma Decision Tree, dan
        tahapan penelitian CRISP-DM yang diimplementasikan pada CDSS DBD.
    </p>
    <div class='hero-tags'>
        <span class='hero-tag'>CRISP-DM</span>
        <span class='hero-tag'>Decision Tree</span>
</div>
""", unsafe_allow_html=True)

col_main, col_side = st.columns([2, 1], gap="large")

with col_main:

    # ── Tujuan Penelitian ──
    st.markdown("<div class='section-header'><h2>🎯 Tujuan Penelitian</h2></div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <p style='font-size:0.9rem; color:#374151; line-height:1.75; margin:0;'>
        Sistem Pendukung Keputusan Klinis (CDSS) ini dikembangkan untuk memfasilitasi tenaga medis
        dalam mengklasifikasikan tingkat keparahan pasien Demam Berdarah Dengue (DBD) menjadi dua
        kategori utama berdasarkan standar <em>International Classification of Diseases 10th Revision (ICD-10)</em>:
        </p>
        <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:1rem; margin-top:1rem;'>
            <div style='background:#EFF6FF; border-radius:12px; padding:1rem 1.1rem;
                        border-left:4px solid #2563EB;'>
                <div style='font-size:0.95rem; font-weight:800; color:#1D4ED8; margin-bottom:0.35rem;'>
                    🔵 A90 — Dengue Fever
                </div>
                <p style='font-size:0.82rem; color:#374151; margin:0; line-height:1.6;'>
                    Kasus DBD klasik tanpa komplikasi perdarahan hebat atau kebocoran plasma.
                    Risiko klinis cenderung lebih rendah.
                </p>
            </div>
            <div style='background:#FEF2F2; border-radius:12px; padding:1rem 1.1rem;
                        border-left:4px solid #DC2626;'>
                <div style='font-size:0.95rem; font-weight:800; color:#DC2626; margin-bottom:0.35rem;'>
                    🔴 A91 — Dengue Hemorrhagic Fever
                </div>
                <p style='font-size:0.82rem; color:#374151; margin:0; line-height:1.6;'>
                    Kasus DBD lebih parah dengan trombositopenia berat dan kebocoran plasma.
                    Memerlukan penanganan dan observasi intensif.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metodologi CRISP-DM ──
    st.markdown("<div class='section-header'><h2>🔄 Metodologi CRISP-DM</h2></div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <p style='font-size:0.875rem; color:#374151; line-height:1.65; margin:0 0 1rem;'>
        Penelitian ini mengadopsi kerangka kerja standar industri data mining, yaitu
        <strong>CRISP-DM</strong> (<em>Cross-Industry Standard Process for Data Mining</em>),
        yang terdiri dari 6 tahapan terstruktur:
        </p>
    """, unsafe_allow_html=True)

    crisp_steps = [
        ("01", "Business Understanding",
         "Mendefinisikan masalah klinis (membedakan A90 dan A91) dan menetapkan tujuan pengembangan CDSS."),
        ("02", "Data Understanding",
         "Mengumpulkan dan menganalisis karakteristik dataset pasien DBD dari RS Aulia (distribusi fitur, missing values, korelasi awal)."),
        ("03", "Data Preparation",
         "Melakukan pra-pemrosesan data: ekstraksi nilai numerik dari string teks, penanganan missing values, dan encoding fitur kategori (Jenis Kelamin)."),
        ("04", "Modeling",
         "Membangun model klasifikasi menggunakan Decision Tree Classifier, dipadukan dengan GridSearchCV untuk pencarian hyperparameter optimal dan penanganan ketidakseimbangan kelas (class weight)."),
        ("05", "Evaluation",
         "Mengevaluasi kinerja model menggunakan metrik statistik (Akurasi, Presisi, Recall, F1-Score) dengan fokus utama pada Recall kelas A91 untuk meminimalkan potensi kesalahan diagnosis pada pasien risiko tinggi."),
        ("06", "Deployment",
         "Mengimplementasikan model ke dalam aplikasi berbasis web (Streamlit) agar dapat digunakan secara praktis oleh tenaga medis."),
    ]

    for num, title, desc in crisp_steps:
        st.markdown(f"""
        <div class='phase-card' style='margin-bottom:0.5rem;'>
            <div class='phase-num'>{num}</div>
            <div class='phase-content'>
                <div class='phase-title'>{title}</div>
                <p class='phase-desc'>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



with col_side:

    # ── Algoritma Decision Tree ──
    st.markdown("<div class='section-header'><h2>🌳 Decision Tree</h2></div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <p style='font-size:0.875rem; color:#374151; line-height:1.7; margin:0 0 0.85rem;'>
        <strong>Decision Tree (Pohon Keputusan)</strong> adalah metode pembelajaran mesin
        <em>supervised</em> yang sangat intuitif. Algoritma ini memecah kumpulan data menjadi
        himpunan bagian yang semakin kecil berdasarkan aturan keputusan berurutan
        (misal: "Apakah Hematokrit &gt; 45?").
        </p>
        <div style='font-size:0.8rem; font-weight:700; color:#6B7280; text-transform:uppercase;
                    letter-spacing:0.06em; margin-bottom:0.6rem;'>Mengapa Decision Tree?</div>
    """, unsafe_allow_html=True)

    dt_reasons = [
        ("🔍", "Transparansi (White-box)", "Alur logika dapat dilacak dan dipahami, sangat krusial dalam domain medis."),
        ("📊", "Feature Importance", "Mampu menghitung signifikansi setiap fitur, memberikan wawasan klinis."),
        ("⚡", "Efisiensi", "Prediksi real-time dari data laboratorium tanpa latensi."),
        ("🔄", "Non-linear Handling", "Menangkap hubungan non-linear kompleks antar variabel medis."),
    ]

    for icon, title, desc in dt_reasons:
        st.markdown(f"""
        <div style='display:flex; gap:0.75rem; align-items:flex-start; margin-bottom:0.65rem;'>
            <span style='font-size:1rem; margin-top:0.1rem;'>{icon}</span>
            <div>
                <div style='font-size:0.82rem; font-weight:700; color:#111827;'>{title}</div>
                <div style='font-size:0.76rem; color:#6B7280; line-height:1.5;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Dataset Info ──
    st.markdown("<br><div class='section-header'><h2>🗄️ Dataset</h2></div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div style='font-size:0.78rem; font-weight:700; color:#6B7280;
                    text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.75rem;'>
            Informasi Dataset
        </div>
    """, unsafe_allow_html=True)

    dataset_info = [
        ("🏥", "Sumber", "RS Aulia"),
        ("👥", "Total Pasien", "~3.000"),
        ("📋", "Kelas Target", "A90 & A91"),
        ("🧪", "Fitur Input", "6 variabel lab"),
        ("📊", "Metodologi Split", "Train/Test split"),
        ("⚖️", "Class Weight", "Balanced"),
    ]

    for icon, label, val in dataset_info:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; align-items:center;
                    padding:0.4rem 0; border-bottom:1px solid #F3F4F6;'>
            <span style='font-size:0.8rem; color:#6B7280;'>{icon} {label}</span>
            <span style='font-size:0.8rem; font-weight:600; color:#111827;'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Feature Importance Note ──
    st.markdown("<br><div class='section-header'><h2>📊 Feature Importance</h2></div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <p style='font-size:0.82rem; color:#374151; line-height:1.7; margin:0;'>
        Skor <strong>Feature Importance</strong> mengukur kontribusi komparatif setiap variabel
        dalam membuat keputusan klasifikasi di dalam struktur pohon.
        <br><br>
        Variabel yang sering digunakan pada simpul-simpul keputusan atas (mendekati akar) dan
        menghasilkan pemisahan data yang paling bersih (penurunan <em>impurity</em> tertinggi)
        akan mendapatkan skor yang lebih tinggi.
        <br><br>
        Di halaman prediksi, sistem memvisualisasikan skor ini untuk memberikan konteks kepada
        tenaga medis mengenai <strong>mengapa model menghasilkan prediksi tertentu</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class='footer-text'>
    &copy; 2026 CDSS DBD RS Aulia &bull; Framework Streamlit &bull; Scikit-Learn Machine Learning
</div>
""", unsafe_allow_html=True)
