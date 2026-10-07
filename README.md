# ⚡ Intelligent Power Grid & Energy Analytics

A collection of Deep Learning models, physics-based simulations, and interactive web tools for power systems fault diagnosis and electricity demand forecasting.

---

## 📌 Projects Overview

### 1. 🌐 Universal Power Grid Fault Classifier (PyTorch)
A deep learning system capable of classifying 16 distinct fault types across **AC 3-Phase**, **AC Single-Phase**, and **Bipolar HVDC** grids.
- **Waveform Physics Simulation**: Simulates AC/DC power transients, voltage dips, fault current surges, third/fifth order harmonics, and Gaussian noise (SNR 30–50 dB).
- **15-Feature Standardized Vector**: Fortescue symmetrical components ($I_0, I_1, I_2, V_0, V_1, V_2$), 3-phase RMS currents & voltages, and transient derivative ($\max(di/dt)$).
- **Universal Multi-Layer Perceptron (MLP)**: Deep neural network with Batch Normalization and Dropout implemented in PyTorch (`universal_fault_model.pth`).
- **High Accuracy**:
  - **100.0%** on AC 3-Phase transmission lines (AG, BG, CG, AB, BC, CA, ABG, BCG, CAG, ABC)
  - **100.0%** on Bipolar HVDC systems (Pole-to-Ground, Pole-to-Pole)
  - **94.4%** across all combined grid topologies

### 2. 🔌 Interactive Fault Classifier Web Dashboard
- Real-time Chart.js waveform visualizer simulating instantaneous fault currents & voltages across 11kV to 765kV systems.
- Live relay trip diagnostics with configurable inception angles, fault resistance, and grid load angles.
- Launch locally: Open `index.html` or run `python3 -m http.server 8080`.

### 3. 📈 ANN Electricity Load Forecasting (TensorFlow / Keras)
- Neural network regression model predicting electrical load demand based on ambient temperature, hour of day, day of week, and historical load.
- Architecture: 4-layer MLP (64 $\rightarrow$ 32 $\rightarrow$ 16 $\rightarrow$ 8 $\rightarrow$ 1) trained for 100 epochs.
- Performance metrics:
  - **$R^2$ Score**: 0.767
  - **MAE**: 9.93 kW
  - **RMSE**: 12.30 kW
- Includes dedicated dashboard in `ann_graphs/index.html`.

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd college

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib joblib tensorflow
```

### 2. Run the Universal Fault Model Evaluation
```bash
python test_ann_model.py
```

### 3. Train the Fault Classification Model
```bash
python fault_classification_ann.py
```

### 4. Launch Web Dashboards
```bash
# Launch Fault Classification Dashboard (Port 8080)
python3 -m http.server 8080

# Launch Load Prediction Dashboard (Port 8000)
cd ann_graphs && python3 -m http.server 8000
```
Open [http://localhost:8080](http://localhost:8080) or [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📂 Repository Structure

```text
├── fault_classification_ann.py   # Universal PyTorch fault model & physics simulator
├── test_ann_model.py             # Evaluation & testing script across 16 fault classes
├── universal_fault_model.pth     # Trained PyTorch weights
├── index.html                    # Universal transmission line fault dashboard
├── ann_graphs/
│   ├── hi.py                     # Electricity load forecasting ANN script
│   ├── index.html                # Electricity load dashboard
│   ├── plot1_load_over_time.png
│   ├── plot2_training_loss.png
│   └── plot3_actual_vs_predicted.png
├── ann_visualizations.py         # Visualizations generator for loss, ROC, and confusion matrices
└── README.md                     # Documentation
```
