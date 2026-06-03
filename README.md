# Coupling Hybrid Deep Learning Prediction with DRL-Based Control for Energy-Efficient Data Center Thermal Management

**Final Year Project (FYP)**

---

## Overview

This project proposes an integrated framework that couples a hybrid **1D CNN-LSTM** temperature prediction model with a **Deep Deterministic Policy Gradient (DDPG)** reinforcement learning control agent for energy-efficient data centre thermal management.

The framework addresses three objectives:
1. Develop a deep learning model to predict return air temperature (T_Return) from historical sensor data
2. Design a cooling control algorithm that uses predicted temperature to optimise energy usage (PUE)
3. Evaluate model performance using standard metrics (MAE, RMSE, R²)

---

## Repository Structure

```
├── CNN-LSTM_baseline_version.ipynb   # Baseline univariate CNN-LSTM model
├── CNN-LSTM_enhanced_version.ipynb   # Enhanced multivariate CNN-LSTM model
├── hvac_ddpg.py                      # CNN-LSTM + DDPG integrated control agent
├── hvac_ddpg_baseline.py             # Standalone DDPG baseline (7D state, no prediction)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Dataset

**TDC2.0 — An Air-Cooled Tropical Data Centre Dataset**
- Source: Le Van Duc (2023), DR-NTU, Nanyang Technological University
- DOI: [10.21979/N9/BLBQ2T](https://doi.org/10.21979/N9/BLBQ2T)
- 521,280 records at 30-second intervals (Sept 2022 – Feb 2023)
- Key features used: `T_Return`, `T_Supply`, `T_Outdoor`, `HVAC_Power_kW`, `RH_Outdoor`, `IT_Power_Total_kW`

> The dataset is not included in this repository due to its size. Download from the DOI link above.

---

## Model Architecture

### Enhanced 1D CNN-LSTM (Prediction)

| Layer | Configuration |
|---|---|
| Conv1D | 64 filters, kernel=5, ReLU, padding=same |
| MaxPooling1D | pool_size=2 |
| Dropout | 0.2 |
| LSTM | 64 units, Tanh activation |
| Dropout | 0.2 |
| Dense | 32 units, ReLU, L2=0.01 |
| Dense | 1 unit (T_Return output) |

- Input: 16 timesteps × 6 features (8-minute lookback at 30-second resolution)
- Optimizer: Adam (lr=0.0005), Loss: MSE
- Callbacks: EarlyStopping (patience=15), ReduceLROnPlateau (patience=5)

### DDPG Control Agent

- **State space (8D):** CNN-LSTM predicted T_Return, T_Outdoor, T_Supply, HVAC Power (COP-estimated), RH_Outdoor, RH_Return, IT_Power, comfort error
- **Action:** T_Supply setpoint mapped to [18.28°C, 30.31°C]
- **Reward:** −PUE penalty − 0.1×comfort error − log-barrier thermal penalty + 0.3×ΔPUE bonus
- **Environment:** First-order thermal lag simulation (α=0.96) calibrated from TDC2

---

## Results

### CNN-LSTM Prediction Performance

| Metric | Baseline CNN-LSTM | Enhanced CNN-LSTM | Improvement |
|---|---|---|---|
| MAE (°C) | 0.1236 | 0.0633 | 48.8% reduction |
| RMSE (°C) | 0.1366 | 0.0809 | 40.8% reduction |
| R² Score | 0.9857 | 0.9950 | +0.0093 |

### DDPG Control Performance (Cross-validated, 5 runs)

| Metric | Baseline DDPG (7D) | CNN-LSTM+DDPG (8D) |
|---|---|---|
| Mean PUE | 1.3965 | 1.4262 |
| Mean Comfort Error (°C) | 3.8437 | 3.1472 |
| Mean Reward | −183.82 | −168.26 |
| Safety Violations | 14 (1 run) | 0 |

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/ha2ardbunny/Hybrid-DL-DRL-Energy-Efficient-Data-Center-Thermal-Management.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/WSL
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

> Tested on: Python 3.11.9 | TensorFlow 2.18.0 | WSL2 Ubuntu with NVIDIA RTX 4050 (CUDA)

---

## References

- Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. *arXiv:1509.02971*
- Le Van Duc. (2023). TDC2: An air-cooled tropical data centre dataset. DR-NTU. https://doi.org/10.21979/N9/BLBQ2T
- muntasirhsn. (2025). CNN-LSTM model for energy usage forecasting. GitHub. https://github.com/muntasirhsn/CNN-LSTM-model-for-energy-usage-forecasting

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
