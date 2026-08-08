"""
pages/2_Evaluasi_Model.py
=========================
Halaman Evaluasi Model - CDSS DBD
Menampilkan metrik performa model Decision Tree

Tahap CRISP-DM: Evaluation
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import joblib

# ── Path Setup ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ── Page Config ──
st.set_page_config(
    page_title="Evaluasi Model | CDSS DBD",
    page_icon="📊",
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


# ── Load Metrics ──
METRICS_PATH = os.path.join(BASE_DIR, "models", "eval_metrics.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline_dbd.pkl")
FALLBACK = os.path.join(BASE_DIR, "models", "decision_tree.pkl")

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
        default_index=2,
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
    elif selected_page == "Tentang Sistem":
        st.switch_page("pages/3_Tentang_Sistem.py")

    model = load_model()
    metrics_sidebar = load_metrics()

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

metrics = load_metrics()

# ── Hero Header ──
st.markdown("""
<div class='hero-card'>
    <div class='hero-badge'>📊 Evaluasi Model</div>
    <h1>Evaluasi Performa Model Decision Tree</h1>
    <p>
        Hasil evaluasi model pada data uji (test set) menggunakan metodologi CRISP-DM fase Evaluation.<br>
        Model dievaluasi dengan <strong>Accuracy, Precision, Recall, F1-Score</strong>,
        Confusion Matrix, dan Classification Report.
    </p>
    <div class='hero-tags'>
        <span class='hero-tag'>Accuracy</span>
        <span class='hero-tag'>Precision</span>
        <span class='hero-tag'>Recall</span>
        <span class='hero-tag'>F1-Score</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Cek Metrics ──
if metrics is None:
    st.error("⚠️ Metrik evaluasi belum tersedia. Jalankan `python train_model.py` terlebih dahulu.")
    st.code("python train_model.py", language="bash")
    st.stop()

# ─────────────────────────────────────────────
#  METRIC CARDS UTAMA
# ─────────────────────────────────────────────

st.markdown("<div class='section-header'><h2>📈 Metrik Evaluasi Keseluruhan</h2></div>",
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    acc = metrics.get('accuracy', 0)
    st.markdown(f"""
    <div class='metric-card blue'>
        <div class='metric-card-icon'>🎯</div>
        <div class='metric-card-value'>{acc*100:.2f}%</div>
        <div class='metric-card-label'>Accuracy</div>
        <div class='metric-card-sub'>Overall accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    prec = metrics.get('precision', 0)
    st.markdown(f"""
    <div class='metric-card green'>
        <div class='metric-card-icon'>🔍</div>
        <div class='metric-card-value'>{prec*100:.2f}%</div>
        <div class='metric-card-label'>Precision</div>
        <div class='metric-card-sub'>Macro average</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    rec = metrics.get('recall', 0)
    st.markdown(f"""
    <div class='metric-card amber'>
        <div class='metric-card-icon'>📡</div>
        <div class='metric-card-value'>{rec*100:.2f}%</div>
        <div class='metric-card-label'>Recall</div>
        <div class='metric-card-sub'>Macro average</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    f1 = metrics.get('f1_score', 0)
    st.markdown(f"""
    <div class='metric-card red'>
        <div class='metric-card-icon'>⚖️</div>
        <div class='metric-card-value'>{f1*100:.2f}%</div>
        <div class='metric-card-label'>F1-Score</div>
        <div class='metric-card-sub'>Macro average</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PENJELASAN PRIORITAS RECALL A91
# ─────────────────────────────────────────────

st.info("""
🎯 **Prioritas Recall Kelas A91 (Dengue Hemorrhagic Fever)**

Penelitian ini memprioritaskan **Recall kelas A91** sebagai metrik utama evaluasi model.
Alasannya adalah:

> Kesalahan **False Negative** pada kelas A91 berarti pasien dengan kondisi 
> **Dengue Hemorrhagic Fever (lebih berat)** akan diprediksi sebagai A90 (Dengue Fever).
> Hal ini berpotensi menyebabkan penanganan yang **kurang tepat** dan dapat membahayakan 
> keselamatan pasien.

Oleh karena itu, model dioptimalkan menggunakan `scoring='recall'` pada GridSearchCV 
dan `class_weight='balanced'` pada DecisionTreeClassifier.
""")

# ─────────────────────────────────────────────
#  METRIK PER KELAS
# ─────────────────────────────────────────────

st.markdown("<div class='section-header'><h2>🏷️ Metrik Per Kelas ICD-10</h2></div>",
            unsafe_allow_html=True)

col_a90, col_a91 = st.columns(2, gap="large")

with col_a90:
    p_a90 = metrics.get('precision_a90', 0)
    r_a90 = metrics.get('recall_a90', 0)
    f_a90 = metrics.get('f1_a90', 0)
    n_a90 = metrics.get('class_counts', {}).get('A90', 0)

    st.markdown(f"""
    <div class='class-card class-card-a90'>
        <div class='class-card-header' style='color:#1D4ED8;'>
            🔵 ICD-10 A90 — Dengue Fever
        </div>
        <div class='class-sub-cards'>
            <div class='class-sub-card'>
                <div class='class-sub-value' style='color:#1D4ED8;'>{p_a90*100:.1f}%</div>
                <div class='class-sub-label'>Precision</div>
            </div>
            <div class='class-sub-card'>
                <div class='class-sub-value' style='color:#1D4ED8;'>{r_a90*100:.1f}%</div>
                <div class='class-sub-label'>Recall</div>
            </div>
            <div class='class-sub-card'>
                <div class='class-sub-value' style='color:#1D4ED8;'>{f_a90*100:.1f}%</div>
                <div class='class-sub-label'>F1-Score</div>
            </div>
        </div>
        <div style='margin-top:0.85rem; font-size:0.78rem; color:#6B7280;'>
            Jumlah sampel test: <strong style='color:#111827;'>{n_a90}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_a91:
    p_a91 = metrics.get('precision_a91', 0)
    r_a91 = metrics.get('recall_a91', 0)
    f_a91 = metrics.get('f1_a91', 0)
    n_a91 = metrics.get('class_counts', {}).get('A91', 0)

    st.markdown(f"""
    <div class='class-card class-card-a91'>
        <div class='class-card-header' style='color:#DC2626;'>
            🔴 ICD-10 A91 — Dengue Hemorrhagic Fever
            <span style='font-size:0.72rem; background:#DC2626; color:white;
                         border-radius:12px; padding:0.12rem 0.5rem; margin-left:0.5rem;
                         font-weight:600;'>★ Prioritas</span>
        </div>
        <div class='class-sub-cards'>
            <div class='class-sub-card'>
                <div class='class-sub-value' style='color:#DC2626;'>{p_a91*100:.1f}%</div>
                <div class='class-sub-label'>Precision</div>
            </div>
            <div class='class-sub-card' style='border:2px solid #FCA5A5;'>
                <div class='class-sub-value' style='color:#DC2626;'>{r_a91*100:.1f}%</div>
                <div class='class-sub-label' style='color:#DC2626;'>Recall ★</div>
            </div>
            <div class='class-sub-card'>
                <div class='class-sub-value' style='color:#DC2626;'>{f_a91*100:.1f}%</div>
                <div class='class-sub-label'>F1-Score</div>
            </div>
        </div>
        <div style='margin-top:0.85rem; font-size:0.78rem; color:#6B7280;'>
            Jumlah sampel test: <strong style='color:#111827;'>{n_a91}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS: CONFUSION MATRIX & CLASSIFICATION REPORT & FEATURE IMPORTANCE
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔲 Confusion Matrix", "📄 Classification Report", "📊 Feature Importance"])

with tab1:
    st.markdown("#### Confusion Matrix — Data Test")
    st.caption(
        "Matriks konfusi menunjukkan jumlah prediksi yang benar (True Positive, True Negative) "
        "dan salah (False Positive, False Negative) untuk setiap kelas."
    )

    cm = np.array(metrics.get('confusion_matrix', [[0, 0], [0, 0]]))

    # Plotly Heatmap
    labels = ['A90 (DF)', 'A91 (DHF)']
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=f"<b>{cm[i][j]}</b>",
                font=dict(
                    color='white' if (i == j and cm[i][j] > cm.max() * 0.3) else '#111827',
                    size=18,
                    family='Inter'
                ),
                showarrow=False
            ))

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, '#DBEAFE'], [0.5, '#2563EB'], [1, '#1D4ED8']],
        showscale=True,
        colorbar=dict(title="Count", tickfont=dict(size=11)),
        hoverongaps=False,
        hovertemplate='Aktual: %{y}<br>Prediksi: %{x}<br>Count: %{z}<extra></extra>',
    ))

    fig_cm.update_layout(
        annotations=annotations,
        xaxis=dict(
            title="<b>Prediksi</b>",
            side='bottom',
            tickfont=dict(size=12, color='#111827'),
            tickangle=0,
            automargin=True
        ),
        yaxis=dict(
            title="<b>Aktual</b>",
            tickfont=dict(size=12, color='#111827'),
            autorange='reversed',
            automargin=True
        ),
        height=420,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=30, b=40),
        font=dict(family='Inter'),
    )

    col_cm1, col_cm2 = st.columns([3, 2], gap="large")
    with col_cm1:
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cm2:
        st.markdown("**Interpretasi Confusion Matrix:**")
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        interp_data = [
            ("✅ True Negative (TN)", tn, "A90 diprediksi benar sebagai A90"),
            ("❌ False Positive (FP)", fp, "A90 diprediksi salah sebagai A91"),
            ("⚠️ False Negative (FN)", fn, "A91 diprediksi salah sebagai A90 ← Kritis!"),
            ("✅ True Positive (TP)", tp, "A91 diprediksi benar sebagai A91"),
        ]
        for label, count, desc in interp_data:
            if "FN" in label:
                bg = "#FEF2F2"
                border = "border-left:3px solid #DC2626;"
            elif "✅" in label:
                bg = "#F0FDF4"
                border = "border-left:3px solid #16A34A;"
            else:
                bg = "#FFFBEB"
                border = "border-left:3px solid #D97706;"

            st.markdown(f"""
            <div style='background:{bg}; border-radius:8px; padding:0.6rem 0.9rem;
                        margin-bottom:0.4rem; {border}'>
                <div style='font-size:0.82rem; font-weight:700; color:#111827;'>
                    {label}: <span style='font-size:1.05rem;'>{count}</span>
                </div>
                <div style='font-size:0.75rem; color:#6B7280; margin-top:0.15rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#F0FDF4; border-radius:8px; padding:0.75rem; margin-top:0.5rem;
                    border-left:3px solid #16A34A;'>
            <div style='font-size:0.8rem; color:#15803D; font-weight:600;'>
                📌 FN (False Negative A91) = <strong>{fn}</strong><br>
                Pasien A91 yang terlewat = berpotensi bahaya klinis
            </div>
        </div>
        """, unsafe_allow_html=True)


with tab2:
    st.markdown("#### Classification Report — Data Test")
    st.caption("Laporan klasifikasi lengkap per kelas beserta nilai rata-rata macro dan weighted.")

    cr = metrics.get('classification_report', {})

    if cr and isinstance(cr, dict):
        # Buat DataFrame dari classification report
        report_rows = []
        class_map   = {'0': 'A90 (Dengue Fever)', '1': 'A91 (DHF)'}

        for key in ['0', '1']:
            if key in cr:
                row = cr[key]
                report_rows.append({
                    'Kelas'    : class_map.get(key, key),
                    'Precision': f"{row.get('precision', 0):.4f}",
                    'Recall'   : f"{row.get('recall', 0):.4f}",
                    'F1-Score' : f"{row.get('f1-score', 0):.4f}",
                    'Support'  : int(row.get('support', 0)),
                })

        for key in ['macro avg', 'weighted avg']:
            if key in cr:
                row = cr[key]
                report_rows.append({
                    'Kelas'    : key.title(),
                    'Precision': f"{row.get('precision', 0):.4f}",
                    'Recall'   : f"{row.get('recall', 0):.4f}",
                    'F1-Score' : f"{row.get('f1-score', 0):.4f}",
                    'Support'  : int(row.get('support', 0)),
                })

        df_report = pd.DataFrame(report_rows)
        st.dataframe(
            df_report.set_index("Kelas"),
            use_container_width=True,
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.4f"),
                "Recall"   : st.column_config.NumberColumn(format="%.4f"),
                "F1-Score" : st.column_config.NumberColumn(format="%.4f"),
            }
        )

    # Raw text report
    with st.expander("📄 Lihat Classification Report (Format Teks)"):
        cr_text = metrics.get('classification_report_text', '')
        st.code(cr_text, language="text")

    # Training vs Test Accuracy
    train_acc = metrics.get('train_accuracy', 0)
    test_acc  = metrics.get('accuracy', 0)
    overfitting_gap = abs(train_acc - test_acc)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📊 Training vs Testing Accuracy:**")
    col_tt1, col_tt2, col_tt3 = st.columns(3)

    with col_tt1:
        st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
    with col_tt2:
        st.metric("Testing Accuracy", f"{test_acc*100:.2f}%")
    with col_tt3:
        gap_color = "inverse" if overfitting_gap > 0.1 else "normal"
        st.metric(
            "Overfitting Gap",
            f"{overfitting_gap*100:.2f}%",
            delta=f"{'Perlu Perhatian' if overfitting_gap > 0.1 else 'OK'}",
            delta_color=gap_color
        )


with tab3:
    st.markdown("#### Feature Importance — Decision Tree")
    st.caption(
        "Tingkat kepentingan setiap variabel input dalam keputusan klasifikasi model. "
        "Nilai lebih tinggi = variabel lebih berpengaruh."
    )

    fi_data = metrics.get('feature_importances', None)
    fi_names = metrics.get('feature_names', [
        'Usia', 'Jenis Kelamin', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit'
    ])

    if fi_data:
        fi_arr = np.array(fi_data)
        sorted_idx = np.argsort(fi_arr)[::-1]
        names_s = [fi_names[i] for i in sorted_idx]
        vals_s  = [fi_arr[i] for i in sorted_idx]

        col_fi1, col_fi2 = st.columns([2, 3], gap="large")

        with col_fi1:
            st.markdown("**Ranking Variabel:**")
            colors_bar = ["#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"]
            for i, (nm, vl) in enumerate(zip(names_s, vals_s)):
                pct = vl * 100
                col = colors_bar[min(i, len(colors_bar)-1)]
                st.markdown(f"""
                <div style='margin-bottom:0.65rem;'>
                    <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;'>
                        <div style='background:{col}; color:white; width:24px; height:24px; border-radius:50%;
                                    font-size:0.72rem; font-weight:700; display:inline-flex;
                                    align-items:center; justify-content:center;'>{i+1}</div>
                        <div style='font-weight:600; font-size:0.88rem; color:#111827; flex:1;'>{nm}</div>
                        <div style='font-weight:700; font-size:0.88rem; color:{col};'>{pct:.2f}%</div>
                    </div>
                    <div style='background:#E5E7EB; border-radius:100px; height:10px; overflow:hidden;'>
                        <div style='width:{pct:.1f}%; height:100%; background:{col}; border-radius:100px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_fi2:
            # Horizontal Bar Chart
            fig_fi2 = go.Figure(go.Bar(
                x=vals_s,
                y=names_s,
                orientation='h',
                marker=dict(
                    color=vals_s,
                    colorscale=[[0, '#BFDBFE'], [0.3, '#60A5FA'], [0.7, '#2563EB'], [1, '#1D4ED8']],
                    line=dict(color='rgba(0,0,0,0)', width=0),
                ),
                text=[f"  {v:.4f}" for v in vals_s],
                textposition='outside',
                textfont=dict(size=13, color='#111827', family='Inter'),
                hovertemplate='%{y}: <b>%{x:.4f}</b><extra></extra>',
            ))
            fig_fi2.update_layout(
                title=dict(
                    text="Feature Importance Score",
                    font=dict(size=15, color='#111827', family='Inter'),
                    x=0.01
                ),
                xaxis=dict(
                    title="Importance Score",
                    showgrid=True,
                    gridcolor='#F3F4F6',
                    range=[0, max(vals_s) * 1.25],
                    tickfont=dict(size=11),
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=13, color='#111827', family='Inter'),
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=80, t=50, b=40),
                height=320,
            )
            st.plotly_chart(fig_fi2, use_container_width=True)

        # Interpretasi Feature Importance
        top_feature = names_s[0]
        top_val = vals_s[0] * 100
        st.info(f"""
        📌 **Variabel paling berpengaruh:** **{top_feature}** ({top_val:.2f}%)

        Berdasarkan feature importance model Decision Tree, **{top_feature}** 
        memberikan kontribusi terbesar dalam membedakan pasien A90 dan A91.
        Hal ini konsisten dengan literatur klinis dimana perubahan nilai laboratorium 
        tersebut menjadi indikator utama keparahan infeksi dengue.
        """)
    else:
        st.warning("Feature importance tidak tersedia.")

# Footer
st.markdown("""
<div class='footer-text'>
    CRISP-DM Fase Evaluation &bull; Decision Tree Classifier &bull; Dataset RS Aulia &bull; CDSS DBD v1.0
</div>
""", unsafe_allow_html=True)
