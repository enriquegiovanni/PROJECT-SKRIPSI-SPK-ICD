"""
plot_tree_light.py
==================
Ilustrasi struktur pemisahan Decision Tree — Light Theme
Load model dari pipeline_dbd.pkl yang sudah tersimpan.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings, joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import _tree

warnings.filterwarnings('ignore')

# ── Load model pipeline ──────────────────────────────────────
MODEL_PATH = r'dbd_clinical_decision_support\models\pipeline_dbd.pkl'
pipeline   = joblib.load(MODEL_PATH)
model      = pipeline.named_steps['classifier']
print(f"Model loaded — Depth={model.get_depth()} | Leaves={model.get_n_leaves()}")

FEATURE_NAMES = ['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit', 'Jenis Kelamin']

# ── Extract tree (4 level) ────────────────────────────────────
tree_ = model.tree_
MAX_DEPTH_SHOW = 4

def get_nodes(node=0, depth=0):
    if depth > MAX_DEPTH_SHOW:
        return None
    is_leaf = (tree_.children_left[node] == _tree.TREE_LEAF)
    vals    = tree_.value[node][0]
    info = {
        'id':        node,
        'depth':     depth,
        'is_leaf':   is_leaf,
        'feature':   FEATURE_NAMES[tree_.feature[node]] if not is_leaf else None,
        'threshold': tree_.threshold[node]               if not is_leaf else None,
        'samples':   int(tree_.n_node_samples[node]),
        'values':    [int(v) for v in vals],
        'dominant':  int(np.argmax(vals)),
        'gini':      round(tree_.impurity[node], 3),
        'left':      None,
        'right':     None,
    }
    if not is_leaf:
        info['left']  = get_nodes(tree_.children_left[node],  depth + 1)
        info['right'] = get_nodes(tree_.children_right[node], depth + 1)
    return info

root = get_nodes()

# ── X-position layout ─────────────────────────────────────────
node_x = {}
W      = 2 ** MAX_DEPTH_SHOW   # 16 unit lebar

def assign_x(node, lo, hi):
    if node is None: return
    node_x[node['id']] = (lo + hi) / 2
    if node['left']:  assign_x(node['left'],  lo, (lo+hi)/2)
    if node['right']: assign_x(node['right'], (lo+hi)/2, hi)

assign_x(root, 0, W)

# ── Light Color Palette ───────────────────────────────────────
BG      = '#F8FAFC'      # canvas putih kebiruan
GRID_C  = '#E2E8F0'      # garis level

# Split node
SPLIT_BG = '#EFF6FF'     # biru sangat muda
SPLIT_BD = '#2563EB'     # biru

# Leaf A90 — biru
A90_BG  = '#DBEAFE'
A90_BD  = '#1D4ED8'
A90_TXT = '#1E40AF'

# Leaf A91 — merah
A91_BG  = '#FEE2E2'
A91_BD  = '#DC2626'
A91_TXT = '#991B1B'

EDGE_C  = '#94A3B8'      # warna anak panah
YES_C   = '#2563EB'      # label Ya
NO_C    = '#DC2626'      # label Tidak
TXT_C   = '#1E293B'      # teks utama
SUB_C   = '#64748B'      # teks kecil

LEVEL_Y  = {0: 10.5, 1: 8.5, 2: 6.5, 3: 4.5, 4: 2.5}
NODE_W, NODE_H = 2.1, 1.0
LEAF_W, LEAF_H = 2.0, 1.0

# ── Figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(44, 18), dpi=130)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(-1.5, W + 1.5)
ax.set_ylim(1.0, 12.5)
ax.axis('off')

# Garis horisontal level
for lvl, ly in LEVEL_Y.items():
    ax.axhline(ly, color=GRID_C, lw=1.2, ls='--', alpha=0.7, zorder=0)

# ── Draw Function ─────────────────────────────────────────────
def draw_node(node):
    if node is None: return
    x  = node_x[node['id']]
    y  = LEVEL_Y[node['depth']]

    if node['is_leaf']:
        is_a90 = node['dominant'] == 0
        bg, bd, tc = (A90_BG, A90_BD, A90_TXT) if is_a90 else (A91_BG, A91_BD, A91_TXT)
        label  = 'A90\nDemam Dengue' if is_a90 else 'A91\nDemam Berdarah'
        w, h   = LEAF_W, LEAF_H

        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle='round,pad=0.1', facecolor=bg,
            edgecolor=bd, linewidth=2.2, zorder=3)
        ax.add_patch(box)

        ax.text(x, y + 0.18, label,
            ha='center', va='center', fontsize=8.5,
            fontweight='bold', color=tc, zorder=4, multialignment='center')
        ax.text(x, y - 0.28,
            f"n={node['samples']}  [{node['values'][0]}, {node['values'][1]}]",
            ha='center', va='center', fontsize=7, color=SUB_C, zorder=4)

    else:
        feat = node['feature']
        thr  = node['threshold']
        if feat == 'Jenis Kelamin':
            thr_str = '≤ 0.5  (Perempuan)'
        elif thr >= 1000:
            thr_str = f'≤ {thr:,.1f}'
        else:
            thr_str = f'≤ {thr:.3f}'

        w, h = NODE_W, NODE_H
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle='round,pad=0.1', facecolor=SPLIT_BG,
            edgecolor=SPLIT_BD, linewidth=2.2, zorder=3)
        ax.add_patch(box)

        ax.text(x, y + 0.27, feat,
            ha='center', va='center', fontsize=9,
            fontweight='bold', color='#1D4ED8', zorder=4)
        ax.text(x, y + 0.02, thr_str,
            ha='center', va='center', fontsize=8, color=TXT_C, zorder=4)
        ax.text(x, y - 0.28,
            f"n={node['samples']}   Gini={node['gini']}",
            ha='center', va='center', fontsize=7, color=SUB_C, zorder=4)

    # Edges to children
    for child, is_left in [(node.get('left'), True), (node.get('right'), False)]:
        if child:
            cx   = node_x[child['id']]
            cy   = LEVEL_Y[child['depth']]
            w_c  = LEAF_W if child['is_leaf'] else NODE_W
            h_c  = LEAF_H if child['is_leaf'] else NODE_H
            h_p  = NODE_H

            y_start = y - h_p / 2 - 0.02
            y_end   = cy + h_c / 2 + 0.02

            ax.annotate('',
                xy=(cx, y_end), xytext=(x, y_start),
                arrowprops=dict(
                    arrowstyle='->', color=EDGE_C,
                    lw=1.6, connectionstyle='arc3,rad=0.0'
                ), zorder=2)

            mid_x = (x + cx) / 2
            mid_y = (y_start + y_end) / 2
            lbl   = 'Ya ✓' if is_left else 'Tidak ✗'
            ec    = YES_C  if is_left else NO_C

            ax.text(mid_x, mid_y, lbl,
                ha='center', va='center', fontsize=7.5,
                fontweight='bold', color=ec, zorder=5,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor=ec,
                          linewidth=1.2, alpha=0.95))

    if node.get('left'):  draw_node(node['left'])
    if node.get('right'): draw_node(node['right'])

draw_node(root)

# ── Level Labels (kiri) ───────────────────────────────────────
level_info = {
    0: ('ROOT', '#1D4ED8'),
    1: ('Level 1', '#334155'),
    2: ('Level 2', '#334155'),
    3: ('Level 3', '#334155'),
    4: ('Daun / Leaf', '#6B21A8'),
}
for lvl, (lbl, clr) in level_info.items():
    ax.text(-1.2, LEVEL_Y[lvl], lbl,
        ha='center', va='center', fontsize=8.5,
        fontweight='bold', color=clr,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=GRID_C, linewidth=1))

# ── Legend (kanan atas) ───────────────────────────────────────
lx, ly = W + 1.1, 11.5
for i, (bg, bd, tc, lbl) in enumerate([
    (SPLIT_BG, SPLIT_BD, '#1D4ED8', 'Node Pemisahan (Split)'),
    (A90_BG,   A90_BD,   A90_TXT,   'A90 — Demam Dengue (DD)'),
    (A91_BG,   A91_BD,   A91_TXT,   'A91 — Demam Berdarah (DBD)'),
]):
    yy = ly - i * 0.75
    b  = FancyBboxPatch((lx - 1.15, yy - 0.27), 2.3, 0.54,
        boxstyle='round,pad=0.07', facecolor=bg, edgecolor=bd, linewidth=1.8)
    ax.add_patch(b)
    ax.text(lx, yy, lbl, ha='center', va='center',
        fontsize=8, fontweight='bold', color=tc)

# Keterangan nilai node
ax.text(lx, ly - 2.6,
    "Keterangan Node:\n"
    "Baris 1 : Fitur & Threshold\n"
    "Baris 2 : n = jumlah sampel\n"
    "Baris 3 : Gini impurity\n"
    "[x, y]  : jumlah [A90, A91]",
    ha='center', va='top', fontsize=7.5, color=SUB_C,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
              edgecolor=GRID_C, linewidth=1))

# ── Title ─────────────────────────────────────────────────────
ax.set_title(
    'Ilustrasi Struktur Pemisahan Decision Tree — Klasifikasi Kode ICD DBD\n'
    f'Decision Tree + Class Weight  |  Total Kedalaman: {model.get_depth()}  |  '
    f'Total Daun: {model.get_n_leaves()}  |  Ditampilkan: 4 Level Pertama',
    fontsize=14, fontweight='bold', color=TXT_C, pad=20,
    fontfamily='DejaVu Sans'
)

plt.tight_layout(pad=1.5)
out = 'decision_tree_light.png'
plt.savefig(out, bbox_inches='tight', facecolor=BG, dpi=150)
plt.close()
print(f"Saved: {out}")
