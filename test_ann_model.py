"""
===========================================================================
Test Script: Verify the Universal ANN Fault Model Predicts Correctly
===========================================================================
Loads the saved PyTorch model + scaler and runs fresh simulations to
confirm prediction quality across all 16 fault classes and 3 grid types.
===========================================================================
"""

import numpy as np
import joblib
import torch
from fault_classification_ann import (
    UniversalRelayModel, UniversalGridSimulator, SignalProcessor,
    FAULT_TYPES, FAULT_TO_IDX, IDX_TO_FAULT, VOLTAGE_LEVELS_KV
)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── 1. Load saved model & scaler ──────────────────────────────────────────
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

model = UniversalRelayModel(input_dim=15, num_classes=16).to(device)
model.load_state_dict(torch.load("universal_fault_model.pth", map_location=device))
model.eval()

scaler = joblib.load("universal_scaler.pkl")

print(f"✅ Model loaded on device: {device}")
print(f"✅ Scaler loaded (fitted on {scaler.n_features_in_} features)\n")

# ── 2. Generate fresh test data (never seen by the model) ─────────────────
np.random.seed(999)  # Different seed from training (42)

# Define which faults belong to which grid
GRID_FAULTS = {
    'DC':     ['NF', 'P_G', 'N_G', 'P_N'],
    'AC_1PH': ['NF', 'L1_G', 'L1_N'],
    'AC_3PH': ['NF', 'AG', 'BG', 'CG', 'AB', 'BC', 'CA', 'ABG', 'BCG', 'CAG', 'ABC'],
}

DC_VOLTAGES   = [400.0, 500.0, 800.0]
AC1_VOLTAGES  = [11.0, 33.0]
AC3_VOLTAGES  = VOLTAGE_LEVELS_KV

SAMPLES_PER_FAULT = 200  # 200 fresh samples per fault class

tasks = []
for grid, faults in GRID_FAULTS.items():
    voltages = DC_VOLTAGES if grid == 'DC' else AC1_VOLTAGES if grid == 'AC_1PH' else AC3_VOLTAGES
    for f in faults:
        for _ in range(SAMPLES_PER_FAULT):
            v = float(np.random.choice(voltages))
            snr = np.random.uniform(30, 50)
            tasks.append((grid, f, v, snr))

print(f"🔬 Generating {len(tasks):,} fresh test samples...")

X_test, y_test = [], []
for args in tasks:
    features, label = UniversalGridSimulator.generate_fault(*args)
    X_test.append(features)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"   Feature matrix shape: {X_test.shape}")
print(f"   Label array shape:    {y_test.shape}\n")

# ── 3. Predict ─────────────────────────────────────────────────────────────
X_test_scaled = scaler.transform(X_test)
X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    outputs = model(X_tensor)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

# ── 4. Report results ─────────────────────────────────────────────────────
overall_acc = accuracy_score(y_test, preds)

print("=" * 70)
print(f"  OVERALL TEST ACCURACY: {overall_acc * 100:.2f}%")
print("=" * 70)

present_classes = np.unique(y_test)
target_names = [IDX_TO_FAULT[i] for i in present_classes]

print("\n📊 Per-Class Classification Report:\n")
print(classification_report(
    y_test, preds,
    labels=present_classes,
    target_names=target_names,
    digits=4
))

# ── 5. Quick per-grid-type breakdown ───────────────────────────────────────
print("\n📈 Accuracy by Grid Type:")
print("-" * 40)
for grid, faults in GRID_FAULTS.items():
    indices = [i for i, label in enumerate(y_test) if IDX_TO_FAULT[label] in faults]
    if indices:
        grid_acc = accuracy_score(y_test[indices], preds[indices])
        print(f"  {grid:8s}  →  {grid_acc * 100:.2f}%  ({len(indices)} samples)")
print()

# ── 6. Show some individual prediction examples ───────────────────────────
print("🔍 Sample Predictions (first 5 per grid type):")
print("-" * 55)
shown = {g: 0 for g in GRID_FAULTS}
for i in range(len(y_test)):
    true_name = IDX_TO_FAULT[y_test[i]]
    pred_name = IDX_TO_FAULT[preds[i]]
    grid = [g for g, fs in GRID_FAULTS.items() if true_name in fs][0]
    if shown[grid] < 5:
        match = "✅" if true_name == pred_name else "❌"
        print(f"  {match}  Grid={grid:8s}  True={true_name:5s}  Predicted={pred_name:5s}")
        shown[grid] += 1
    if all(v >= 5 for v in shown.values()):
        break

print("\n✅ Test complete.")
