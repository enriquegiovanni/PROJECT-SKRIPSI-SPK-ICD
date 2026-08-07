"""
plot_tree_ilustrasi.py
======================
Membuat ilustrasi struktur pemisahan Decision Tree yang bersih dan informatif
menggunakan matplotlib dengan custom node styling.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

# ── Parsing ───────────────────────────────────────────────────
def parse_usia(t):
    if pd.isna(t): return np.nan
    t = str(t).strip()
    m = re.search(r'(\d+)\s*(Th|tahun)', t, re.IGNORECASE)
    if m: return float(m.group(1))
    m = re.search(r'(\d+)\s*(Bl|bulan)', t, re.IGNORECASE)
    return 0.0 if m else np.nan

def parse_lab(t):
    if pd.isna(t): return np.nan
    m = re.search(r'([\d]+\.?[\d]*)', str(t).strip())
    return float(m.group(1)) if m else np.nan

# ── Preprocessing ─────────────────────────────────────────────
print("Memuat & preprocessing data...")
df_raw = pd.read_excel('Data_Lab_Penyakit_DBD_RS_Aulia.xlsx')
cols = {
    'Usia (tahun)': 'Usia',
    'Jenis Kelamin\n(L/P)': 'Jenis_Kelamin',
    'Trombosit\n(ribu/μL)': 'Trombosit',
    'Hematokrit (%)': 'Hematokrit',
    'Hemoglobin (g/dL)': 'Hemoglobin',
    'Leukosit\n(ribu/μL)': 'Leukosit',
    'kode ICD': 'ICD'
}
df = df_raw[list(cols.keys())].rename(columns=cols).copy()
df['Usia'] = df['Usia'].apply(parse_usia)
for c in ['Trombosit','Hematokrit','Hemoglobin','Leukosit']:
    df[c] = df[c].apply(parse_lab)
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].astype(str).str.strip().str.upper()
df['ICD'] = df['ICD'].astype(str).str.strip().str.upper()
df[['Usia','Trombosit','Hematokrit','Hemoglobin','Leukosit']] = \
    SimpleImputer(strategy='median').fit_transform(df[['Usia','Trombosit','Hematokrit','Hemoglobin','Leukosit']])
df[['Jenis_Kelamin']] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Jenis_Kelamin']])
df = df.drop_duplicates(subset=['Usia','Jenis_Kelamin','Trombosit','Hematokrit','Hemoglobin','Leukosit','ICD']).reset_index(drop=True)
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].replace({'LAKI-LAKI':'L','LAKI LAKI':'L','PRIA':'L','M':'L','PEREMPUAN':'P','WANITA':'P','F':'P'})
df = df[df['Jenis_Kelamin'].isin(['L','P'])].reset_index(drop=True)
df = df[df['ICD'].isin(['A90','A91'])].copy()
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].map({'L':1,'P':0})
df['ICD'] = df['ICD'].map({'A91':1,'A90':0})

X = df[['Usia','Trombosit','Hematokrit','Hemoglobin','Leukosit','Jenis_Kelamin']]
y = df['ICD'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
smote = SMOTE(random_state=RANDOM_STATE)
X_tr_s, y_tr_s = smote.fit_resample(X_train, y_train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("GridSearchCV...")
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    {'criterion':['gini','entropy'],'max_depth':[3,5,7,10,15,None],
     'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4,8],'class_weight':['balanced',None]},
    scoring='recall_macro', cv=cv, n_jobs=-1
)
gs.fit(X_train, y_train)
model = gs.best_estimator_
model.fit(X_train, y_train)
print(f"Depth={model.get_depth()} | Leaves={model.get_n_leaves()} | Recall={recall_score(y_test, model.predict(X_test), average='macro'):.4f}")

FEATURE_NAMES = ['Usia', 'Trombosit', 'Hematokrit', 'Hemoglobin', 'Leukosit', 'Jenis Kelamin']

# ── Extract tree structure (max 4 levels untuk ilustrasi) ─────
tree_ = model.tree_
MAX_DEPTH_SHOW = 4  # Batasi 4 level agar terbaca

def get_tree_nodes(node=0, depth=0):
    """Rekursif ekstrak node info."""
    if depth > MAX_DEPTH_SHOW:
        return None
    is_leaf = (tree_.children_left[node] == _tree.TREE_LEAF)
    vals = tree_.value[node][0]
    total = vals.sum()
    dominant = int(np.argmax(vals))
    gini = tree_.impurity[node]
    
    info = {
        'id': node,
        'depth': depth,
        'is_leaf': is_leaf,
        'feature': FEATURE_NAMES[tree_.feature[node]] if not is_leaf else None,
        'threshold': tree_.threshold[node] if not is_leaf else None,
        'samples': int(tree_.n_node_samples[node]),
        'values': [int(v) for v in vals],
        'dominant': dominant,
        'gini': round(gini, 3),
        'left': None,
        'right': None,
    }
    if not is_leaf:
        info['left']  = get_tree_nodes(tree_.children_left[node],  depth+1)
        info['right'] = get_tree_nodes(tree_.children_right[node], depth+1)
    return info

root = get_tree_nodes()

# ── Layout Computation ────────────────────────────────────────
node_positions = {}

def assign_x(node, left_bound, right_bound):
    if node is None: return
    mid = (left_bound + right_bound) / 2
    node_positions[node['id']] = mid
    if node['left']:
        assign_x(node['left'],  left_bound, mid)
    if node['right']:
        assign_x(node['right'], mid, right_bound)

total_leaves = 2 ** MAX_DEPTH_SHOW
assign_x(root, 0, total_leaves)

# ── Draw ──────────────────────────────────────────────────────
BG     = '#0F172A'
A90_C  = '#1D4ED8'   # biru gelap
A91_C  = '#B91C1C'   # merah gelap
A90_L  = '#3B82F6'   # biru terang
A91_L  = '#EF4444'   # merah terang
SPLIT_C = '#1E3A5F'  # biru node split
EDGE_C  = '#64748B'  # warna garis
TEXT_C  = '#F1F5F9'
SUBTEXT = '#94A3B8'

LEVEL_Y  = {0: 10.0, 1: 8.0, 2: 6.0, 3: 4.0, 4: 2.0}
NODE_W   = 2.0
NODE_H   = 0.9
LEAF_W   = 1.9
LEAF_H   = 1.0

fig, ax = plt.subplots(figsize=(42, 18), dpi=120)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(-1, total_leaves + 1)
ax.set_ylim(0.5, 11.5)
ax.axis('off')

def get_node_color(node):
    if node['is_leaf']:
        return (A90_C, A90_L) if node['dominant'] == 0 else (A91_C, A91_L)
    return ('#1E3A5F', '#3B82F6')

def draw_node(ax, node):
    if node is None: return
    x  = node_positions[node['id']]
    y  = LEVEL_Y[node['depth']]
    bg, border = get_node_color(node)

    if node['is_leaf']:
        label   = 'A90\nDemam Dengue' if node['dominant'] == 0 else 'A91\nDemam Berdarah'
        txt_col = A90_L if node['dominant'] == 0 else A91_L
        w, h    = LEAF_W, LEAF_H

        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle='round,pad=0.08',
            facecolor=bg, edgecolor=border, linewidth=2.0,
            zorder=3
        )
        ax.add_patch(box)

        ax.text(x, y + 0.18, label, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=txt_col, zorder=4,
                multialignment='center')
        info = f"n={node['samples']}  [{node['values'][0]}, {node['values'][1]}]"
        ax.text(x, y - 0.25, info, ha='center', va='center',
                fontsize=6.5, color=SUBTEXT, zorder=4)
    else:
        # Format threshold
        feat = node['feature']
        thr  = node['threshold']
        if feat == 'Jenis Kelamin':
            thr_str = '≤ 0.5\n(Perempuan)'
        elif thr >= 1000:
            thr_str = f'≤ {thr:,.0f}'
        else:
            thr_str = f'≤ {thr:.2f}'
        
        w, h = NODE_W, NODE_H
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle='round,pad=0.08',
            facecolor='#0B2545', edgecolor='#2563EB', linewidth=1.8,
            zorder=3
        )
        ax.add_patch(box)

        # Feature name
        ax.text(x, y + 0.25, feat, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='#93C5FD', zorder=4)
        # Threshold
        ax.text(x, y - 0.05, thr_str, ha='center', va='center',
                fontsize=7.5, color=TEXT_C, zorder=4, multialignment='center')
        # Samples
        ax.text(x, y - 0.32, f'n = {node["samples"]}  |  Gini = {node["gini"]}',
                ha='center', va='center', fontsize=6.5, color=SUBTEXT, zorder=4)

    # Draw edges to children
    for child, side in [(node.get('left'), 'Ya (True)'), (node.get('right'), 'Tidak (False)')]:
        if child:
            cx = node_positions[child['id']]
            cy_start = y - h/2
            cy_end   = LEVEL_Y[child['depth']] + LEAF_H/2 if child['is_leaf'] else LEVEL_Y[child['depth']] + NODE_H/2

            ax.annotate(
                '', xy=(cx, cy_end), xytext=(x, cy_start),
                arrowprops=dict(arrowstyle='->', color=EDGE_C, lw=1.4),
                zorder=2
            )
            mid_x = (x + cx) / 2
            mid_y = (cy_start + cy_end) / 2
            edge_label = side
            edge_col   = '#22D3EE' if side.startswith('Ya') else '#FB923C'
            ax.text(mid_x, mid_y, edge_label,
                    ha='center', va='center', fontsize=6.5,
                    color=edge_col, fontweight='bold', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=BG, edgecolor='none', alpha=0.8))

    if node.get('left'):  draw_node(ax, node['left'])
    if node.get('right'): draw_node(ax, node['right'])

draw_node(ax, root)

# ── Level Labels ──────────────────────────────────────────────
level_labels = {0: 'ROOT\n(Node Akar)', 1: 'Level 1', 2: 'Level 2', 3: 'Level 3', 4: f'Level 4\n(Daun / Leaf)'}
for lvl, lbl in level_labels.items():
    ax.text(-0.6, LEVEL_Y[lvl], lbl,
            ha='center', va='center', fontsize=8, color='#64748B',
            fontstyle='italic', fontweight='bold')

ax.axvline(x=-0.1, color='#1E293B', lw=1, ls='--', alpha=0.5)

# ── Legend ────────────────────────────────────────────────────
legend_x, legend_y = total_leaves - 0.2, 10.8

# A90
b90 = FancyBboxPatch((legend_x - 1.1, legend_y - 0.25), 2.2, 0.5,
    boxstyle='round,pad=0.07', facecolor=A90_C, edgecolor=A90_L, linewidth=1.5)
ax.add_patch(b90)
ax.text(legend_x, legend_y, 'A90 — Demam Dengue (DD)',
    ha='center', va='center', fontsize=8, color=A90_L, fontweight='bold')

# A91
b91 = FancyBboxPatch((legend_x - 1.1, legend_y - 0.95), 2.2, 0.5,
    boxstyle='round,pad=0.07', facecolor=A91_C, edgecolor=A91_L, linewidth=1.5)
ax.add_patch(b91)
ax.text(legend_x, legend_y - 0.7, 'A91 — Demam Berdarah (DBD)',
    ha='center', va='center', fontsize=8, color=A91_L, fontweight='bold')

# Split node
bsp = FancyBboxPatch((legend_x - 1.1, legend_y - 1.65), 2.2, 0.5,
    boxstyle='round,pad=0.07', facecolor='#0B2545', edgecolor='#2563EB', linewidth=1.5)
ax.add_patch(bsp)
ax.text(legend_x, legend_y - 1.4, 'Node Pemisahan (Split)',
    ha='center', va='center', fontsize=8, color='#93C5FD', fontweight='bold')

# ── Title ─────────────────────────────────────────────────────
ax.set_title(
    'Ilustrasi Struktur Pemisahan Decision Tree — Klasifikasi Kode ICD DBD\n'
    f'Model: Decision Tree + Class Weight  |  Total Depth: {model.get_depth()}  |  '
    f'Leaf Nodes: {model.get_n_leaves()}  |  Ditampilkan: 4 Level Pertama',
    fontsize=14, fontweight='bold', color='white', pad=18
)

plt.tight_layout(pad=1.5)
out = 'decision_tree_ilustrasi.png'
plt.savefig(out, bbox_inches='tight', facecolor=BG, dpi=150)
plt.close()
print(f"Saved: {out}")
