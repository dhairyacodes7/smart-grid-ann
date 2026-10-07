"""
===================================================================================
Universal Power Grid Fault Classifier (AC 3-Phase, AC 1-Phase, HVDC)
===================================================================================
Features:
1. Unified Physics Engine: Simulates AC 3-Phase, AC Single-Phase, and Bipolar HVDC.
2. Universal 15-Feature Vector Extraction (handles Fortescue, RMS, and di/dt transients).
3. 16-Class Universal PyTorch Model (detects any fault on any grid topology).
===================================================================================
"""

import math
import logging
import numpy as np
import pandas as pd
import joblib
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ===================================================================================
# CONFIGURATION & CONSTANTS
# ===================================================================================
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Universal Fault Dictionary (16 Classes)
FAULT_TYPES = [
    'NF',                                               # Universal Normal
    'AG', 'BG', 'CG', 'AB', 'BC', 'CA',                 # AC 3-Phase
    'ABG', 'BCG', 'CAG', 'ABC',                         # AC 3-Phase
    'L1_G', 'L1_N',                                     # AC 1-Phase
    'P_G', 'N_G', 'P_N'                                 # HVDC (Pos-Gnd, Neg-Gnd, Pole-to-Pole)
]
FAULT_TO_IDX = {f: i for i, f in enumerate(FAULT_TYPES)}
IDX_TO_FAULT = {i: f for i, f in enumerate(FAULT_TYPES)}

GRID_TYPES = {'DC': 0, 'AC_1PH': 1, 'AC_3PH': 3}
VOLTAGE_LEVELS_KV = [11.0, 33.0, 66.0, 132.0, 220.0, 400.0, 765.0, 800.0] # Added 800kV for HVDC
FREQ = 50.0
OMEGA = 2.0 * np.pi * FREQ
SAMPLES_PER_CYCLE = 64
N_CYCLES = 2
TOTAL_SAMPLES = SAMPLES_PER_CYCLE * N_CYCLES
DT = 1.0 / (FREQ * SAMPLES_PER_CYCLE)


# ===================================================================================
# 1. SIGNAL PROCESSOR CLASS (Universal 15-Feature Extractor)
# ===================================================================================
class SignalProcessor:
    @staticmethod
    def compute_symmetrical_components(ia: complex, ib: complex, ic: complex) -> Tuple[float, float, float]:
        a = np.exp(1j * 2.0 * np.pi / 3.0)
        a_sq = a ** 2
        i0 = abs((1.0 / 3.0) * (ia + ib + ic))
        i1 = abs((1.0 / 3.0) * (ia + a * ib + a_sq * ic))
        i2 = abs((1.0 / 3.0) * (ia + a_sq * ib + a * ic))
        return i0, i1, i2

    @staticmethod
    def extract_universal_features(grid_type: str, v_level_kv: float, t: np.ndarray, 
                                   v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, 
                                   i1: np.ndarray, i2: np.ndarray, i3: np.ndarray) -> np.ndarray:
        """
        Extracts 15 standardized features regardless of grid type.
        [grid_type_code, v_level_kv, V1rms, V2rms, V3rms, I1rms, I2rms, I3rms, V0, V1, V2, I0, I1, I2, max_di_dt]
        """
        grid_code = GRID_TYPES[grid_type]
        
        # 1. RMS Calculations
        v1_rms = np.sqrt(np.mean(v1**2)); i1_rms = np.sqrt(np.mean(i1**2))
        v2_rms = np.sqrt(np.mean(v2**2)); i2_rms = np.sqrt(np.mean(i2**2))
        v3_rms = np.sqrt(np.mean(v3**2)); i3_rms = np.sqrt(np.mean(i3**2))

        # 2. Transient Analysis (Crucial for DC faults)
        di1_dt = np.max(np.abs(np.gradient(i1, DT)))
        di2_dt = np.max(np.abs(np.gradient(i2, DT)))
        di3_dt = np.max(np.abs(np.gradient(i3, DT)))
        max_di_dt = max(di1_dt, di2_dt, di3_dt)

        # 3. Symmetrical Components (Only valid for AC_3PH)
        v0 = v1_seq = v2_seq = i0 = i1_seq = i2_seq = 0.0
        
        if grid_type == 'AC_3PH':
            N = len(t)
            fund_idx = np.argmin(np.abs(np.fft.fftfreq(N, d=DT) - FREQ))
            v1_ph = (np.fft.fft(v1) / N)[fund_idx] * 2.0
            v2_ph = (np.fft.fft(v2) / N)[fund_idx] * 2.0
            v3_ph = (np.fft.fft(v3) / N)[fund_idx] * 2.0
            i1_ph = (np.fft.fft(i1) / N)[fund_idx] * 2.0
            i2_ph = (np.fft.fft(i2) / N)[fund_idx] * 2.0
            i3_ph = (np.fft.fft(i3) / N)[fund_idx] * 2.0

            v0, v1_seq, v2_seq = SignalProcessor.compute_symmetrical_components(v1_ph, v2_ph, v3_ph)
            i0, i1_seq, i2_seq = SignalProcessor.compute_symmetrical_components(i1_ph, i2_ph, i3_ph)

        return np.array([
            grid_code, v_level_kv, 
            v1_rms, v2_rms, v3_rms, 
            i1_rms, i2_rms, i3_rms, 
            v0, v1_seq, v2_seq, 
            i0, i1_seq, i2_seq, 
            max_di_dt
        ])


# ===================================================================================
# 2. UNIVERSAL TRANSMISSION GRID SIMULATOR
# ===================================================================================
class UniversalGridSimulator:
    @staticmethod
    def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
        if snr_db >= 100: return signal
        noise_power = np.mean(signal ** 2) / (10 ** (snr_db / 10.0))
        return signal + np.random.normal(0, np.sqrt(noise_power), size=signal.shape)

    @staticmethod
    def add_harmonics(signal: np.ndarray, t: np.ndarray, thd_pct: float) -> np.ndarray:
        fundamental_mag = np.max(np.abs(signal))
        h3 = (thd_pct / 100.0) * fundamental_mag * 0.7 * np.sin(3 * OMEGA * t)
        h5 = (thd_pct / 100.0) * fundamental_mag * 0.3 * np.sin(5 * OMEGA * t)
        return signal + h3 + h5

    @staticmethod
    def generate_fault(grid_type: str, fault_type: str, v_level_kv: float, 
                       snr_db: float) -> Tuple[np.ndarray, int]:
        t = np.arange(0, TOTAL_SAMPLES) * DT
        v_nom = (v_level_kv * 1000.0)
        i_nom = 800.0 if v_level_kv < 400 else 1500.0

        v1 = v2 = v3 = np.zeros_like(t)
        i1 = i2 = i3 = np.zeros_like(t)

        fault_mask = t >= (N_CYCLES / 2.0 / FREQ)
        t_f = t[fault_mask]
        
        # Fault physics parameters
        dip = np.random.uniform(0.1, 0.3)
        spike = np.random.uniform(5.0, 15.0) * i_nom

        # ==========================================
        # HVDC GRID SIMULATION (Bipolar)
        # ==========================================
        if grid_type == 'DC':
            v1 = np.ones_like(t) * (v_nom / 2.0)   # Positive Pole
            v2 = np.ones_like(t) * (-v_nom / 2.0)  # Negative Pole
            i1 = np.ones_like(t) * i_nom
            i2 = np.ones_like(t) * -i_nom
            
            if fault_type == 'P_G':
                v1[fault_mask] *= dip
                i1[fault_mask] += spike * (1 - np.exp(-500 * (t_f - t_f[0]))) # DC transient exponential
            elif fault_type == 'N_G':
                v2[fault_mask] *= dip
                i2[fault_mask] -= spike * (1 - np.exp(-500 * (t_f - t_f[0])))
            elif fault_type == 'P_N':
                v1[fault_mask] *= dip
                v2[fault_mask] *= dip
                i1[fault_mask] += spike * (1 - np.exp(-500 * (t_f - t_f[0])))
                i2[fault_mask] -= spike * (1 - np.exp(-500 * (t_f - t_f[0])))

        # ==========================================
        # AC 1-PHASE GRID SIMULATION
        # ==========================================
        elif grid_type == 'AC_1PH':
            v1 = np.sqrt(2) * (v_nom / np.sqrt(3)) * np.sin(OMEGA * t) # Live Phase
            v2 = np.zeros_like(t)                                      # Neutral
            i1 = np.sqrt(2) * i_nom * np.sin(OMEGA * t - 0.5)
            
            if fault_type in ['L1_G', 'L1_N']:
                v1[fault_mask] *= dip
                i1[fault_mask] += np.sqrt(2) * spike * np.sin(OMEGA * t_f - np.pi/4)

        # ==========================================
        # AC 3-PHASE GRID SIMULATION
        # ==========================================
        elif grid_type == 'AC_3PH':
            v_nom_ph = v_nom / np.sqrt(3)
            v1 = np.sqrt(2) * v_nom_ph * np.sin(OMEGA * t)
            v2 = np.sqrt(2) * v_nom_ph * np.sin(OMEGA * t - 2*np.pi/3)
            v3 = np.sqrt(2) * v_nom_ph * np.sin(OMEGA * t + 2*np.pi/3)

            i1 = np.sqrt(2) * i_nom * np.sin(OMEGA * t - 0.5)
            i2 = np.sqrt(2) * i_nom * np.sin(OMEGA * t - 2*np.pi/3 - 0.5)
            i3 = np.sqrt(2) * i_nom * np.sin(OMEGA * t + 2*np.pi/3 - 0.5)

            if fault_type != 'NF':
                if 'A' in fault_type: v1[fault_mask] *= dip
                if 'B' in fault_type: v2[fault_mask] *= dip
                if 'C' in fault_type: v3[fault_mask] *= dip

                if fault_type == 'AG': i1[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
                elif fault_type == 'BG': i2[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
                elif fault_type == 'CG': i3[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
                elif fault_type == 'AB': 
                    diff = np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
                    i1[fault_mask] += diff; i2[fault_mask] -= diff
                elif fault_type == 'BC': 
                    diff = np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
                    i2[fault_mask] += diff; i3[fault_mask] -= diff
                elif fault_type == 'CA': 
                    diff = np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
                    i3[fault_mask] += diff; i1[fault_mask] -= diff
                elif fault_type == 'ABG':
                    i1[fault_mask] += np.sqrt(2)*spike*0.9*np.sin(OMEGA*t_f - np.pi/4)
                    i2[fault_mask] += np.sqrt(2)*spike*0.7*np.sin(OMEGA*t_f - np.pi/3)
                elif fault_type == 'BCG':
                    i2[fault_mask] += np.sqrt(2)*spike*0.9*np.sin(OMEGA*t_f - np.pi/4)
                    i3[fault_mask] += np.sqrt(2)*spike*0.7*np.sin(OMEGA*t_f - np.pi/3)
                elif fault_type == 'CAG':
                    i3[fault_mask] += np.sqrt(2)*spike*0.9*np.sin(OMEGA*t_f - np.pi/4)
                    i1[fault_mask] += np.sqrt(2)*spike*0.7*np.sin(OMEGA*t_f - np.pi/3)
                elif fault_type == 'ABC':
                    i1[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
                    i2[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)
                    i3[fault_mask] += np.sqrt(2)*spike*np.sin(OMEGA*t_f - np.pi/4)

            # Apply Harmonics (AC only)
            thd = np.random.uniform(0.0, 5.0)
            v1 = UniversalGridSimulator.add_harmonics(v1, t, thd)
            v2 = UniversalGridSimulator.add_harmonics(v2, t, thd)
            v3 = UniversalGridSimulator.add_harmonics(v3, t, thd)
            i1 = UniversalGridSimulator.add_harmonics(i1, t, thd)
            i2 = UniversalGridSimulator.add_harmonics(i2, t, thd)
            i3 = UniversalGridSimulator.add_harmonics(i3, t, thd)

        # Apply Universal Noise
        v1 = UniversalGridSimulator.add_noise(v1, snr_db)
        v2 = UniversalGridSimulator.add_noise(v2, snr_db)
        v3 = UniversalGridSimulator.add_noise(v3, snr_db)
        i1 = UniversalGridSimulator.add_noise(i1, snr_db)
        i2 = UniversalGridSimulator.add_noise(i2, snr_db)
        i3 = UniversalGridSimulator.add_noise(i3, snr_db)

        features = SignalProcessor.extract_universal_features(grid_type, v_level_kv, t, v1, v2, v3, i1, i2, i3)
        return features, FAULT_TO_IDX[fault_type]


# ===================================================================================
# 3. UNIVERSAL PYTORCH NEURAL NETWORK
# ===================================================================================
class UniversalRelayModel(nn.Module):
    """Deep MLP adapted for Universal 15-Feature Input and 16-Class Output."""
    def __init__(self, input_dim: int = 15, num_classes: int = 16):
        super(UniversalRelayModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ===================================================================================
# 4. TRAINING & MAIN EXECUTION
# ===================================================================================
def _worker(args):
    return UniversalGridSimulator.generate_fault(*args)

def main():
    logger.info("=================================================================")
    logger.info(" UNIVERSAL POWER GRID FAULT CLASSIFIER (HVDC, AC-1PH, AC-3PH) ")
    logger.info("=================================================================")

    tasks = []
    np.random.seed(42)
    
    # Generate balanced dataset per Grid Type
    SAMPLES_PER_GRID = 50000 
    
    # 1. HVDC Dataset
    dc_faults = ['NF', 'P_G', 'N_G', 'P_N']
    for f in dc_faults:
        for _ in range(SAMPLES_PER_GRID // len(dc_faults)):
            v = float(np.random.choice([400.0, 500.0, 800.0])) # HVDC standard levels
            tasks.append(('DC', f, v, np.random.uniform(30, 50)))
            
    # 2. AC 1-Phase Dataset
    ac1_faults = ['NF', 'L1_G', 'L1_N']
    for f in ac1_faults:
        for _ in range(SAMPLES_PER_GRID // len(ac1_faults)):
            v = float(np.random.choice([11.0, 33.0])) # Common distribution levels
            tasks.append(('AC_1PH', f, v, np.random.uniform(30, 50)))

    # 3. AC 3-Phase Dataset
    ac3_faults = ['NF', 'AG', 'BG', 'CG', 'AB', 'BC', 'CA', 'ABG', 'BCG', 'CAG', 'ABC']
    for f in ac3_faults:
        for _ in range(SAMPLES_PER_GRID // len(ac3_faults)):
            v = float(np.random.choice(VOLTAGE_LEVELS_KV))
            tasks.append(('AC_3PH', f, v, np.random.uniform(30, 50)))

    logger.info(f"Executing Universal Simulation across {len(tasks):,} grid scenarios...")
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_worker, tasks, chunksize=1000))

    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    logger.info(f"Dataset generated. Feature matrix: {X.shape}, Target array: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "universal_scaler.pkl")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = UniversalRelayModel(input_dim=15, num_classes=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=256, shuffle=False)

    logger.info(f"Training Universal AI Model on {device}...")
    for epoch in range(1, 41): # 40 Epochs limit for demo speed
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
                correct += (torch.max(outputs, 1)[1] == batch_y).sum().item()
                
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch [{epoch:02d}/40] - Train Loss: {train_loss/len(X_train):.4f} | Val Acc: {correct/len(X_test)*100:.2f}%")

    torch.save(model.state_dict(), "universal_fault_model.pth")
    
    model.eval()
    with torch.no_grad():
        preds = torch.max(model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device)), 1)[1].cpu().numpy()

    acc = accuracy_score(y_test, preds)
    logger.info("=================================================================")
    logger.info(f" FINAL UNIVERSAL MODEL TEST ACCURACY: {acc * 100:.2f}%")
    logger.info("=================================================================")
    
    # We must filter out classes that weren't sampled (e.g. if len(tasks) didn't cleanly divide)
    present_classes = np.unique(y_test)
    target_names = [IDX_TO_FAULT[i] for i in present_classes]
    print("\nClassification Report:\n", classification_report(y_test, preds, labels=present_classes, target_names=target_names, digits=4))


if __name__ == '__main__':
    main()
