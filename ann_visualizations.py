"""
===================================================================================
Complete Visualization Suite for Universal Power Grid Fault Classifier ANN
===================================================================================
Generates all graphs related to the ANN model:
  1. Training & Validation Loss Curves
  2. Training & Validation Accuracy Curves
  3. Confusion Matrix (Heatmap)
  4. Per-Class Accuracy Bar Chart
  5. Per-Class Precision / Recall / F1 Bar Chart
  6. Dataset Class Distribution
  7. Feature Correlation Heatmap
  8. Feature Importance (Permutation-based)
  9. ROC Curves (One-vs-Rest, all 16 classes)
 10. Sample Fault Waveforms (AC 3-Phase, AC 1-Phase, HVDC)
 11. t-SNE Embedding of Learned Features
 12. Neural Network Architecture Diagram
===================================================================================
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from concurrent.futures import ProcessPoolExecutor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ── import everything from the main ANN file ──
from fault_classification_ann import (
    FAULT_TYPES, FAULT_TO_IDX, IDX_TO_FAULT, GRID_TYPES,
    VOLTAGE_LEVELS_KV, FREQ, OMEGA, SAMPLES_PER_CYCLE, N_CYCLES,
    TOTAL_SAMPLES, DT, SignalProcessor, UniversalGridSimulator,
    UniversalRelayModel, _worker
)

# ── plotting style ──
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.facecolor': 'white',
})

SAVE_DIR = "ann_graphs"
import os
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_CLASSES = len(FAULT_TYPES)


# =====================================================================================
# 1 — DATA GENERATION (identical to main, but smaller for faster graphing; still large)
# =====================================================================================
def generate_dataset(samples_per_grid: int = 15000):
    """Generate a smaller but representative dataset for visualisation."""
    tasks = []
    np.random.seed(42)

    dc_faults = ['NF', 'P_G', 'N_G', 'P_N']
    for f in dc_faults:
        for _ in range(samples_per_grid // len(dc_faults)):
            v = float(np.random.choice([400.0, 500.0, 800.0]))
            tasks.append(('DC', f, v, np.random.uniform(30, 50)))

    ac1_faults = ['NF', 'L1_G', 'L1_N']
    for f in ac1_faults:
        for _ in range(samples_per_grid // len(ac1_faults)):
            v = float(np.random.choice([11.0, 33.0]))
            tasks.append(('AC_1PH', f, v, np.random.uniform(30, 50)))

    ac3_faults = ['NF', 'AG', 'BG', 'CG', 'AB', 'BC', 'CA', 'ABG', 'BCG', 'CAG', 'ABC']
    for f in ac3_faults:
        for _ in range(samples_per_grid // len(ac3_faults)):
            v = float(np.random.choice(VOLTAGE_LEVELS_KV))
            tasks.append(('AC_3PH', f, v, np.random.uniform(30, 50)))

    print(f"[INFO] Simulating {len(tasks):,} grid scenarios …")
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_worker, tasks, chunksize=500))

    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    print(f"[INFO] Dataset ready — X {X.shape}, y {y.shape}")
    return X, y


# =====================================================================================
# 2 — TRAIN MODEL & COLLECT EPOCH-WISE METRICS
# =====================================================================================
def train_model(X, y, epochs: int = 40, batch_size: int = 256, lr: float = 0.003):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    model = UniversalRelayModel(input_dim=15, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * bx.size(0)
            t_correct += (out.argmax(1) == by).sum().item()
            t_total += bx.size(0)

        # ── validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                v_loss += criterion(out, by).item() * bx.size(0)
                v_correct += (out.argmax(1) == by).sum().item()
                v_total += bx.size(0)

        history['train_loss'].append(t_loss / t_total)
        history['val_loss'].append(v_loss / v_total)
        history['train_acc'].append(t_correct / t_total * 100)
        history['val_acc'].append(v_correct / v_total * 100)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs}]  "
                  f"Train Loss {history['train_loss'][-1]:.4f}  "
                  f"Val Acc {history['val_acc'][-1]:.2f}%")

    return model, device, scaler, X_tr, X_te, y_train, y_test, history


# =====================================================================================
# GRAPH  1 — Training & Validation Loss
# =====================================================================================
def plot_loss_curves(history):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'o-', color='#e74c3c', label='Training Loss', markersize=3)
    ax.plot(epochs, history['val_loss'],   's-', color='#2980b9', label='Validation Loss', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training & Validation Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/01_loss_curves.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/01_loss_curves.png")


# =====================================================================================
# GRAPH  2 — Training & Validation Accuracy
# =====================================================================================
def plot_accuracy_curves(history):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(history['train_acc']) + 1)
    ax.plot(epochs, history['train_acc'], 'o-', color='#27ae60', label='Training Accuracy', markersize=3)
    ax.plot(epochs, history['val_acc'],   's-', color='#8e44ad', label='Validation Accuracy', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training & Validation Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/02_accuracy_curves.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/02_accuracy_curves.png")


# =====================================================================================
# GRAPH  3 — Confusion Matrix
# =====================================================================================
def plot_confusion_matrix(y_test, preds, present_classes):
    labels = [IDX_TO_FAULT[i] for i in present_classes]
    cm = confusion_matrix(y_test, preds, labels=present_classes)
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(cm_pct, interpolation='nearest', cmap='YlOrRd')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Prediction Rate (%)')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (% per True Class)')

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if cm_pct[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm_pct[i,j]:.1f}', ha='center', va='center',
                    fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/03_confusion_matrix.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/03_confusion_matrix.png")


# =====================================================================================
# GRAPH  4 — Per-Class Accuracy
# =====================================================================================
def plot_per_class_accuracy(y_test, preds, present_classes):
    labels = [IDX_TO_FAULT[i] for i in present_classes]
    cm = confusion_matrix(y_test, preds, labels=present_classes)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(labels)))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, per_class_acc, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=np.mean(per_class_acc), color='red', linestyle='--', linewidth=1,
               label=f'Mean Accuracy ({np.mean(per_class_acc):.1f}%)')
    ax.set_xlabel('Fault Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Classification Accuracy')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/04_per_class_accuracy.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/04_per_class_accuracy.png")


# =====================================================================================
# GRAPH  5 — Per-Class Precision / Recall / F1
# =====================================================================================
def plot_precision_recall_f1(y_test, preds, present_classes):
    labels = [IDX_TO_FAULT[i] for i in present_classes]
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, preds, labels=present_classes, average=None)

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, p * 100, w, label='Precision', color='#3498db')
    ax.bar(x,     r * 100, w, label='Recall',    color='#e67e22')
    ax.bar(x + w, f1 * 100, w, label='F1-Score', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Score (%)')
    ax.set_title('Per-Class Precision, Recall & F1-Score')
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/05_precision_recall_f1.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/05_precision_recall_f1.png")


# =====================================================================================
# GRAPH  6 — Dataset Class Distribution
# =====================================================================================
def plot_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    labels = [IDX_TO_FAULT[i] for i in unique]
    # Grouped colors by grid type
    grid_colors = {'DC': '#e74c3c', 'AC_1PH': '#3498db', 'AC_3PH': '#27ae60'}
    bar_colors = []
    for lbl in labels:
        if lbl in ['P_G', 'N_G', 'P_N']:
            bar_colors.append(grid_colors['DC'])
        elif lbl in ['L1_G', 'L1_N']:
            bar_colors.append(grid_colors['AC_1PH'])
        elif lbl == 'NF':
            bar_colors.append('#7f8c8d')
        else:
            bar_colors.append(grid_colors['AC_3PH'])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, counts, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Fault Type')
    ax.set_ylabel('Sample Count')
    ax.set_title('Dataset Class Distribution')
    ax.grid(axis='y', alpha=0.3)
    patches = [mpatches.Patch(color=c, label=l) for l, c in
               [('Normal', '#7f8c8d'), ('HVDC', '#e74c3c'),
                ('AC 1-Phase', '#3498db'), ('AC 3-Phase', '#27ae60')]]
    ax.legend(handles=patches, loc='upper right')
    for i, (lbl, cnt) in enumerate(zip(labels, counts)):
        ax.text(i, cnt + max(counts)*0.01, str(cnt), ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/06_class_distribution.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/06_class_distribution.png")


# =====================================================================================
# GRAPH  7 — Feature Correlation Heatmap
# =====================================================================================
def plot_feature_correlation(X):
    feature_names = [
        'GridCode', 'V_level', 'V1rms', 'V2rms', 'V3rms',
        'I1rms', 'I2rms', 'I3rms', 'V0', 'V1seq', 'V2seq',
        'I0', 'I1seq', 'I2seq', 'max_di/dt'
    ]
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation')
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title('Feature Correlation Heatmap (15 Universal Features)')
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            color = 'white' if abs(corr.values[i, j]) > 0.6 else 'black'
            ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center',
                    fontsize=5.5, color=color)
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/07_feature_correlation.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/07_feature_correlation.png")


# =====================================================================================
# GRAPH  8 — Feature Importance (Weight-based)
# =====================================================================================
def plot_feature_importance(model, device):
    feature_names = [
        'GridCode', 'V_level', 'V1rms', 'V2rms', 'V3rms',
        'I1rms', 'I2rms', 'I3rms', 'V0', 'V1seq', 'V2seq',
        'I0', 'I1seq', 'I2seq', 'max_di/dt'
    ]
    # Use first-layer weight magnitudes as a proxy for feature importance
    first_layer_weight = list(model.parameters())[0].detach().cpu().numpy()  # shape: (256, 15)
    importance = np.mean(np.abs(first_layer_weight), axis=0)
    importance = importance / importance.sum() * 100  # normalize to %

    sorted_idx = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = importance[sorted_idx]

    colors = plt.cm.magma(np.linspace(0.3, 0.85, len(sorted_names)))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_names)), sorted_vals[::-1], color=colors[::-1], edgecolor='white')
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel('Relative Importance (%)')
    ax.set_title('Feature Importance (First-Layer Weight Magnitude)')
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(sorted_vals[::-1]):
        ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/08_feature_importance.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/08_feature_importance.png")


# =====================================================================================
# GRAPH  9 — ROC Curves (One-vs-Rest)
# =====================================================================================
def plot_roc_curves(model, device, X_te, y_test, present_classes):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax

    y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(present_classes)))

    for idx, cls_idx in enumerate(present_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], probs[:, cls_idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cmap[idx], linewidth=1.2,
                label=f'{IDX_TO_FAULT[cls_idx]} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — One-vs-Rest (All 16 Classes)')
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/09_roc_curves.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/09_roc_curves.png")


# =====================================================================================
# GRAPH 10 — Sample Fault Waveforms
# =====================================================================================
def plot_sample_waveforms():
    t = np.arange(0, TOTAL_SAMPLES) * DT * 1000  # ms

    scenarios = [
        ('AC_3PH', 'AG',  220.0, 'AC 3φ: Phase-A to Ground (AG)'),
        ('AC_3PH', 'ABC', 220.0, 'AC 3φ: Three-Phase (ABC)'),
        ('AC_1PH', 'L1_G', 33.0, 'AC 1φ: Line to Ground (L1_G)'),
        ('DC',     'P_N', 800.0, 'HVDC: Pole-to-Pole (P_N)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, (grid, fault, vlev, title) in zip(axes, scenarios):
        np.random.seed(7)
        sim = UniversalGridSimulator
        t_arr = np.arange(0, TOTAL_SAMPLES) * DT
        v_nom = vlev * 1000.0
        i_nom = 800.0 if vlev < 400 else 1500.0

        # Quick inline sim to get raw waveforms (not features)
        v1 = v2 = v3 = np.zeros_like(t_arr)
        i1 = i2 = i3 = np.zeros_like(t_arr)
        fault_mask = t_arr >= (N_CYCLES / 2.0 / FREQ)
        t_f = t_arr[fault_mask]
        dip = 0.2
        spike = 10.0 * i_nom

        if grid == 'DC':
            v1 = np.ones_like(t_arr) * (v_nom / 2.0)
            v2 = np.ones_like(t_arr) * (-v_nom / 2.0)
            i1 = np.ones_like(t_arr) * i_nom
            i2 = np.ones_like(t_arr) * -i_nom
            if fault == 'P_N':
                v1[fault_mask] *= dip; v2[fault_mask] *= dip
                i1[fault_mask] += spike * (1 - np.exp(-500*(t_f-t_f[0])))
                i2[fault_mask] -= spike * (1 - np.exp(-500*(t_f-t_f[0])))
        elif grid == 'AC_1PH':
            v1 = np.sqrt(2)*(v_nom/np.sqrt(3))*np.sin(OMEGA*t_arr)
            i1 = np.sqrt(2)*i_nom*np.sin(OMEGA*t_arr - 0.5)
            if fault in ['L1_G', 'L1_N']:
                v1[fault_mask] *= dip
                i1[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
        elif grid == 'AC_3PH':
            v_ph = v_nom / np.sqrt(3)
            v1 = np.sqrt(2)*v_ph*np.sin(OMEGA*t_arr)
            v2 = np.sqrt(2)*v_ph*np.sin(OMEGA*t_arr - 2*np.pi/3)
            v3 = np.sqrt(2)*v_ph*np.sin(OMEGA*t_arr + 2*np.pi/3)
            i1 = np.sqrt(2)*i_nom*np.sin(OMEGA*t_arr - 0.5)
            i2 = np.sqrt(2)*i_nom*np.sin(OMEGA*t_arr - 2*np.pi/3 - 0.5)
            i3 = np.sqrt(2)*i_nom*np.sin(OMEGA*t_arr + 2*np.pi/3 - 0.5)
            if fault == 'AG':
                v1[fault_mask] *= dip
                i1[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
            elif fault == 'ABC':
                i1[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
                i2[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
                i3[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)

        # Plot currents on top axes
        ax.plot(t, i1/1000, label='I1', linewidth=0.8)
        if grid != 'AC_1PH':
            ax.plot(t, i2/1000, label='I2', linewidth=0.8)
        if grid == 'AC_3PH':
            ax.plot(t, i3/1000, label='I3', linewidth=0.8)
        ax.axvline(x=(N_CYCLES/2.0/FREQ)*1000, color='red', linestyle='--',
                   linewidth=0.8, label='Fault Onset')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Current (kA)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sample Fault Waveforms Across Grid Topologies', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/10_sample_waveforms.png", bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/10_sample_waveforms.png")


# =====================================================================================
# GRAPH 11 — t-SNE Embedding of Learned Features
# =====================================================================================
def plot_tsne(model, device, X_te, y_test, present_classes, n_samples=3000):
    model.eval()

    # Extract penultimate layer embeddings
    hook_out = []
    def hook_fn(module, inp, out):
        hook_out.append(out.detach().cpu().numpy())

    # The ReLU before the final linear layer (layer index -2 in sequential → the nn.ReLU at position -2)
    handle = model.network[-2].register_forward_hook(hook_fn)

    subset = np.random.choice(len(X_te), min(n_samples, len(X_te)), replace=False)
    X_sub = torch.tensor(X_te[subset], dtype=torch.float32).to(device)
    y_sub = y_test[subset]

    with torch.no_grad():
        _ = model(X_sub)

    handle.remove()
    embeddings = hook_out[0]

    print("  Running t-SNE (this may take a moment)…")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    Z = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    for cls_idx in present_classes:
        mask = y_sub == cls_idx
        if mask.sum() > 0:
            ax.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.6, color=cmap[cls_idx],
                       label=IDX_TO_FAULT[cls_idx])
    ax.set_title('t-SNE Visualization of Penultimate-Layer Embeddings')
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.legend(fontsize=7, markerscale=3, loc='best', ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/11_tsne_embeddings.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/11_tsne_embeddings.png")


# =====================================================================================
# GRAPH 12 — Neural Network Architecture Diagram
# =====================================================================================
def plot_network_architecture():
    layers = [
        ('Input\n(15)', 15),
        ('Dense\n256+BN\nReLU', 256),
        ('Dropout\n0.15', 256),
        ('Dense\n128+BN\nReLU', 128),
        ('Dropout\n0.1', 128),
        ('Dense\n64+BN\nReLU', 64),
        ('Dense\n(16)', 16),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-1, len(layers))
    ax.set_ylim(-3, 3)
    ax.axis('off')
    ax.set_title('ANN Architecture — UniversalRelayModel', fontsize=14, fontweight='bold', pad=20)

    colors = ['#3498db', '#2ecc71', '#e67e22', '#2ecc71', '#e67e22', '#2ecc71', '#e74c3c']
    max_n = max(n for _, n in layers)

    for i, ((label, n), color) in enumerate(zip(layers, colors)):
        # Draw box proportional to layer width
        h = 0.5 + 2.2 * (n / max_n)
        rect = mpatches.FancyBboxPatch((i - 0.35, -h/2), 0.7, h,
                                        boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor='white',
                                        alpha=0.85, linewidth=2)
        ax.add_patch(rect)
        ax.text(i, 0, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
        ax.text(i, -h/2 - 0.25, f'n={n}', ha='center', va='top', fontsize=7, color='#555')

        # Arrow to next layer
        if i < len(layers) - 1:
            ax.annotate('', xy=(i+0.65, 0), xytext=(i+0.35, 0),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/12_network_architecture.png")
    plt.close(fig)
    print(f"  ✅ Saved → {SAVE_DIR}/12_network_architecture.png")


# =====================================================================================
# MAIN ENTRY POINT
# =====================================================================================
def main():
    print("=" * 70)
    print("  ANN VISUALIZATION SUITE — Universal Power Grid Fault Classifier")
    print("=" * 70)

    # 1. Generate data
    X, y = generate_dataset(samples_per_grid=15000)

    # 2. Train model & collect metrics
    model, device, scaler, X_tr, X_te, y_train, y_test, history = train_model(X, y, epochs=40)

    # 3. Get predictions
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
    present_classes = np.unique(y_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n  Final Test Accuracy: {acc*100:.2f}%\n")

    # 4. Generate ALL graphs
    print("─" * 50)
    print("  Generating Graphs …")
    print("─" * 50)

    plot_loss_curves(history)                               # 1
    plot_accuracy_curves(history)                           # 2
    plot_confusion_matrix(y_test, preds, present_classes)   # 3
    plot_per_class_accuracy(y_test, preds, present_classes) # 4
    plot_precision_recall_f1(y_test, preds, present_classes)# 5
    plot_class_distribution(y)                              # 6
    plot_feature_correlation(X)                             # 7
    plot_feature_importance(model, device)                  # 8
    plot_roc_curves(model, device, X_te, y_test, present_classes)  # 9
    plot_sample_waveforms()                                 # 10
    plot_tsne(model, device, X_te, y_test, present_classes) # 11
    plot_network_architecture()                             # 12

    print("\n" + "=" * 70)
    print(f"  ✅ ALL 12 GRAPHS SAVED TO → ./{SAVE_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
