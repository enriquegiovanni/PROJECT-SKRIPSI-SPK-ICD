"""
plot_hasil_model.py
===================
Visualisasi lengkap semua hasil evaluasi model Decision Tree DBD
Light theme — siap untuk laporan/skripsi
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ── Load Metrics ──────────────────────────────────────────────
m = joblib.load(r'dbd_clinical_decision_support\models\eval_metrics.pkl')

# ── Warna & Style ─────────────────────────────────────────────
BG      = '#FFFFFF'
PANEL   = '#F8FAFC'
BORDER  = '#E2E8F0'
TXT     = '#1E293B'
SUB     = '#64748B'
A90_C   = '#2563EB'   # biru
A91_C   = '#DC2626'   # merah
ACC_C   = '#16A34A'   # hijau
WARN_C  = '#D97706'   # oranye

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': PANEL,
    'figure.facecolor': BG,
    'axes.edgecolor': BORDER,
    'axes.labelcolor': TXT,
    'xtick.color': SUB,
    'ytick.color': SUB,
    'text.color': TXT,
    'grid.color': BORDER,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
})

fig = plt.figure(figsize=(20, 24), dpi=130, facecolor=BG)
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    4, 3,
    figure=fig,
    hspace=0.52,
    wspace=0.38,
    left=0.06, right=0.97,
    top=0.94, bottom=0.04
)

# ═══════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════
fig.text(0.5, 0.97, 'Hasil Evaluasi Model Decision Tree — Klasifikasi Kode ICD DBD',
         ha='center', va='top', fontsize=17, fontweight='bold', color=TXT)
fig.text(0.5, 0.955, f"Model: {m['best_model_name']}   |   Data: {m['n_total']} sampel   |   Train: {m['n_train']}   |   Test: {m['n_test']}",
         ha='center', va='top', fontsize=10, color=SUB)


# ═══════════════════════════════════════════════════
#  PANEL 1 — Kartu Metrik Utama (row 0, span 3 col)
# ═══════════════════════════════════════════════════
ax_cards = fig.add_subplot(gs[0, :])
ax_cards.axis('off')
ax_cards.set_xlim(0, 1)
ax_cards.set_ylim(0, 1)
ax_cards.set_facecolor(BG)

cards = [
    ('Accuracy\n(Test)',        f"{m['accuracy']*100:.2f}%",        ACC_C,  '#DCFCE7', 'Akurasi keseluruhan\npada data uji'),
    ('Accuracy\n(Train)',       f"{m['train_accuracy']*100:.2f}%",  '#0891B2','#CFFAFE','Akurasi pada\ndata latih'),
    ('Precision\n(Macro Avg)', f"{m['precision']*100:.2f}%",       A90_C,  '#DBEAFE', 'Rata-rata precision\nkedua kelas'),
    ('Recall\n(Macro Avg)',    f"{m['recall']*100:.2f}%",           '#7C3AED','#EDE9FE','Rata-rata recall\nkedua kelas'),
    ('F1-Score\n(Macro Avg)',  f"{m['f1_score']*100:.2f}%",         WARN_C, '#FEF3C7', 'Harmonic mean\nprecision & recall'),
]

n_cards = len(cards)
card_w  = 0.85 / n_cards
margin  = (1 - card_w * n_cards) / (n_cards + 1)

for i, (title, val, clr, bg, note) in enumerate(cards):
    x0 = margin + i * (card_w + margin)
    box = FancyBboxPatch((x0, 0.08), card_w, 0.82,
        boxstyle='round,pad=0.015', facecolor=bg,
        edgecolor=clr, linewidth=2.2, transform=ax_cards.transAxes,
        clip_on=False, zorder=2)
    ax_cards.add_patch(box)

    cx = x0 + card_w / 2
    ax_cards.text(cx, 0.82, title, ha='center', va='center',
        fontsize=9, fontweight='bold', color=clr, multialignment='center',
        transform=ax_cards.transAxes)
    ax_cards.text(cx, 0.52, val, ha='center', va='center',
        fontsize=20, fontweight='bold', color=clr,
        transform=ax_cards.transAxes)
    ax_cards.text(cx, 0.22, note, ha='center', va='center',
        fontsize=7.5, color=SUB, multialignment='center',
        transform=ax_cards.transAxes)

ax_cards.set_title('Metrik Evaluasi Utama', fontsize=12,
    fontweight='bold', color=TXT, pad=10, loc='left')


# ═══════════════════════════════════════════════════
#  PANEL 2 — Confusion Matrix (row 1, col 0)
# ═══════════════════════════════════════════════════
ax_cm = fig.add_subplot(gs[1, 0])
cm    = np.array(m['confusion_matrix'])
total = cm.sum()

im = ax_cm.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['A90\n(Prediksi)', 'A91\n(Prediksi)'], fontsize=9, color=TXT)
ax_cm.set_yticklabels(['A90\n(Aktual)', 'A91\n(Aktual)'], fontsize=9, color=TXT)
ax_cm.set_title('Confusion Matrix', fontsize=11, fontweight='bold', color=TXT, pad=10)

colors_cm = [['#1D4ED8', '#FFFFFF'], ['#FFFFFF', '#1D4ED8']]
for i in range(2):
    for j in range(2):
        pct = cm[i, j] / total * 100
        ax_cm.text(j, i,
            f'{cm[i,j]}\n({pct:.1f}%)',
            ha='center', va='center',
            fontsize=13, fontweight='bold',
            color='white' if i == j else TXT)

ax_cm.set_xlabel('Prediksi', fontsize=10, fontweight='bold', color=TXT)
ax_cm.set_ylabel('Aktual', fontsize=10, fontweight='bold', color=TXT)

# Anotasi TP/TN/FP/FN
labels_cm = [['TP', 'FN'], ['FP', 'TN']]
for i in range(2):
    for j in range(2):
        ax_cm.text(j + 0.38, i - 0.38, labels_cm[i][j],
            ha='right', va='top', fontsize=7.5,
            color='white' if i == j else '#94A3B8',
            fontstyle='italic')


# ═══════════════════════════════════════════════════
#  PANEL 3 — Metrik Per Kelas (row 1, col 1)
# ═══════════════════════════════════════════════════
ax_pc = fig.add_subplot(gs[1, 1])

metrics_label = ['Precision', 'Recall', 'F1-Score']
a90_vals = [m['precision_a90'], m['recall_a90'], m['f1_a90']]
a91_vals = [m['precision_a91'], m['recall_a91'], m['f1_a91']]

x      = np.arange(len(metrics_label))
width  = 0.32
bars90 = ax_pc.bar(x - width/2, [v*100 for v in a90_vals], width,
    color=A90_C, label='A90 — DD', alpha=0.9, edgecolor='white', linewidth=0.5)
bars91 = ax_pc.bar(x + width/2, [v*100 for v in a91_vals], width,
    color=A91_C, label='A91 — DBD', alpha=0.9, edgecolor='white', linewidth=0.5)

for bars in [bars90, bars91]:
    for bar in bars:
        h = bar.get_height()
        ax_pc.text(bar.get_x() + bar.get_width()/2, h + 1.0,
            f'{h:.1f}%', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=TXT)

ax_pc.set_ylim(0, 115)
ax_pc.set_xticks(x)
ax_pc.set_xticklabels(metrics_label, fontsize=9)
ax_pc.set_ylabel('Nilai (%)', fontsize=9)
ax_pc.set_title('Metrik Per Kelas (A90 vs A91)', fontsize=11, fontweight='bold', color=TXT, pad=10)
ax_pc.legend(fontsize=9, framealpha=0.8, edgecolor=BORDER)
ax_pc.yaxis.grid(True); ax_pc.set_axisbelow(True)
ax_pc.spines[['top','right']].set_visible(False)


# ═══════════════════════════════════════════════════
#  PANEL 4 — Feature Importance (row 1, col 2)
# ═══════════════════════════════════════════════════
ax_fi = fig.add_subplot(gs[1, 2])

fi_names = m['feature_names']
fi_vals  = m['feature_importances']
order    = np.argsort(fi_vals)
fi_names_s = [fi_names[i] for i in order]
fi_vals_s  = [fi_vals[i]  for i in order]

cmap_fi = plt.cm.Blues(np.linspace(0.35, 0.85, len(fi_vals_s)))
bars_fi = ax_fi.barh(fi_names_s, [v*100 for v in fi_vals_s],
    color=cmap_fi, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars_fi, fi_vals_s):
    ax_fi.text(val*100 + 0.5, bar.get_y() + bar.get_height()/2,
        f'{val*100:.2f}%', va='center', fontsize=8.5,
        fontweight='bold', color=TXT)

ax_fi.set_xlim(0, 90)
ax_fi.set_xlabel('Feature Importance (%)', fontsize=9)
ax_fi.set_title('Feature Importance', fontsize=11, fontweight='bold', color=TXT, pad=10)
ax_fi.xaxis.grid(True); ax_fi.set_axisbelow(True)
ax_fi.spines[['top','right']].set_visible(False)


# ═══════════════════════════════════════════════════
#  PANEL 5 — Distribusi Data (row 2, col 0)
# ═══════════════════════════════════════════════════
ax_dist = fig.add_subplot(gs[2, 0])

n_a90 = m['class_counts']['A90']
n_a91 = m['class_counts']['A91']

bars_dist = ax_dist.bar(
    ['A90\nDemam Dengue', 'A91\nDemam Berdarah'],
    [n_a90, n_a91],
    color=[A90_C, A91_C], alpha=0.9,
    edgecolor='white', linewidth=0.5, width=0.5
)

total_test = n_a90 + n_a91
for bar, n in zip(bars_dist, [n_a90, n_a91]):
    pct = n / total_test * 100
    ax_dist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
        f'{n}\n({pct:.1f}%)', ha='center', va='bottom',
        fontsize=10, fontweight='bold', color=TXT)

ax_dist.set_ylim(0, max(n_a90, n_a91) * 1.22)
ax_dist.set_title('Distribusi Data Test per Kelas', fontsize=11, fontweight='bold', color=TXT, pad=10)
ax_dist.set_ylabel('Jumlah Sampel', fontsize=9)
ax_dist.yaxis.grid(True); ax_dist.set_axisbelow(True)
ax_dist.spines[['top','right']].set_visible(False)


# ═══════════════════════════════════════════════════
#  PANEL 6 — Comparison Train vs Test Accuracy (row 2, col 1)
# ═══════════════════════════════════════════════════
ax_trte = fig.add_subplot(gs[2, 1])

labels_tt = ['Train\nAccuracy', 'Test\nAccuracy']
vals_tt   = [m['train_accuracy']*100, m['accuracy']*100]
colors_tt = ['#0891B2', ACC_C]

bars_tt = ax_trte.bar(labels_tt, vals_tt, color=colors_tt,
    alpha=0.9, width=0.4, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars_tt, vals_tt):
    ax_trte.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
        f'{val:.2f}%', ha='center', va='bottom',
        fontsize=13, fontweight='bold', color=TXT)

ax_trte.set_ylim(0, 100)
ax_trte.set_title('Train vs Test Accuracy', fontsize=11, fontweight='bold', color=TXT, pad=10)
ax_trte.set_ylabel('Accuracy (%)', fontsize=9)
ax_trte.yaxis.grid(True); ax_trte.set_axisbelow(True)
ax_trte.spines[['top','right']].set_visible(False)
ax_trte.axhline(y=80, color=WARN_C, ls='--', lw=1.3, label='Baseline 80%')
ax_trte.legend(fontsize=8, framealpha=0.8, edgecolor=BORDER)


# ═══════════════════════════════════════════════════
#  PANEL 7 — Radar / Macro Metrics Summary (row 2, col 2)
# ═══════════════════════════════════════════════════
ax_radar = fig.add_subplot(gs[2, 2])

categories = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)',
              'Precision\nA90', 'Recall\nA90', 'Precision\nA91', 'Recall\nA91']
values = [
    m['accuracy']*100,
    m['precision']*100,
    m['recall']*100,
    m['f1_score']*100,
    m['precision_a90']*100,
    m['recall_a90']*100,
    m['precision_a91']*100,
    m['recall_a91']*100,
]
colors_r = [ACC_C, A90_C, '#7C3AED', WARN_C, A90_C, A90_C, A91_C, A91_C]
bars_r = ax_radar.barh(categories, values, color=colors_r, alpha=0.85,
    edgecolor='white', linewidth=0.5)
for bar, val in zip(bars_r, values):
    ax_radar.text(val + 0.5, bar.get_y() + bar.get_height()/2,
        f'{val:.1f}%', va='center', fontsize=8, fontweight='bold', color=TXT)

ax_radar.set_xlim(0, 110)
ax_radar.set_title('Ringkasan Semua Metrik', fontsize=11, fontweight='bold', color=TXT, pad=10)
ax_radar.set_xlabel('Nilai (%)', fontsize=9)
ax_radar.xaxis.grid(True); ax_radar.set_axisbelow(True)
ax_radar.axvline(x=80, color=WARN_C, ls='--', lw=1.2, label='Baseline 80%')
ax_radar.legend(fontsize=8, framealpha=0.8, edgecolor=BORDER)
ax_radar.spines[['top','right']].set_visible(False)


# ═══════════════════════════════════════════════════
#  PANEL 8 — Classification Report Table (row 3, span 3 col)
# ═══════════════════════════════════════════════════
ax_tbl = fig.add_subplot(gs[3, :])
ax_tbl.axis('off')
ax_tbl.set_title('Classification Report Lengkap', fontsize=12,
    fontweight='bold', color=TXT, pad=10, loc='left')

cr = m['classification_report']
col_labels = ['Kelas', 'Precision', 'Recall', 'F1-Score', 'Support', 'Keterangan']
rows_data = [
    ['A90 — Demam Dengue',
     f"{cr['A90']['precision']*100:.2f}%",
     f"{cr['A90']['recall']*100:.2f}%",
     f"{cr['A90']['f1-score']*100:.2f}%",
     f"{int(cr['A90']['support'])}",
     'Risiko Lebih Rendah'],
    ['A91 — Demam Berdarah',
     f"{cr['A91']['precision']*100:.2f}%",
     f"{cr['A91']['recall']*100:.2f}%",
     f"{cr['A91']['f1-score']*100:.2f}%",
     f"{int(cr['A91']['support'])}",
     'Risiko Lebih Tinggi'],
    ['Macro Avg',
     f"{cr['macro avg']['precision']*100:.2f}%",
     f"{cr['macro avg']['recall']*100:.2f}%",
     f"{cr['macro avg']['f1-score']*100:.2f}%",
     f"{int(cr['macro avg']['support'])}",
     'Rata-rata tiap kelas'],
    ['Weighted Avg',
     f"{cr['weighted avg']['precision']*100:.2f}%",
     f"{cr['weighted avg']['recall']*100:.2f}%",
     f"{cr['weighted avg']['f1-score']*100:.2f}%",
     f"{int(cr['weighted avg']['support'])}",
     'Rata-rata berbobot'],
]

tbl = ax_tbl.table(
    cellText=rows_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0.0, 0.0, 1.0, 1.0]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

# Header style
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#1E293B')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    tbl[(0, j)].set_edgecolor(BORDER)

# Row styles
row_colors = [
    ('#DBEAFE', A90_C),   # A90
    ('#FEE2E2', A91_C),   # A91
    ('#F1F5F9', SUB),     # Macro
    ('#F1F5F9', SUB),     # Weighted
]
for i, (bg, tc) in enumerate(row_colors):
    for j in range(len(col_labels)):
        cell = tbl[(i+1, j)]
        cell.set_facecolor(bg)
        cell.set_edgecolor(BORDER)
        if j == 0:
            cell.set_text_props(fontweight='bold', color=tc)

plt.savefig('hasil_model_lengkap.png', bbox_inches='tight',
            facecolor=BG, dpi=150)
plt.close()
print("Saved: hasil_model_lengkap.png")
