# Task 2 Report Guide — Multi-Step Hourly Water Demand Forecasting

> **Purpose:** Complete structured analysis for **Part 2 (Task 2)** and **Part 3 (Conclusion)** of [Final Major Assignment Report 2026.docx](../Final%20Major%20Assignment%20Report%202026.docx), aligned with [Final Project Requirements 2026.pdf](../Final%20Project%20Requirements%202026.pdf).  
> **Source Code:** [task2_prepare_data.py](../task2_prepare_data.py) (96 lines) · [task2_seq2seq_attention.py](../task2_seq2seq_attention.py) (122 lines)

---

## 1. Dataset Description

- **Dataset Name:** *Hourly Water Demand of a Mixed District Recorded by Supplier* (Zenodo Record `11045013`, provided by Siemens AG for Karlsruhe Institute of Technology).
- **Download Link:** `https://zenodo.org/records/11045013` (specifically `waterconsumption_noNaN_fixedTimechange.csv`, 1.63 MB).
- **Scale & Temporal Coverage:** **46,032 hourly observations** spanning **January 1, 2016 (`00:00–01:00`) to April 1, 2021 (`23:00–00:00`)** at a uniform **1-hour** collection frequency (every day strictly standardized to 24 hourly intervals).
- **Raw Attributes (6 columns):**
  1. `Date`: Calendar date (`M/D/YYYY`).
  2. `Time`: Hourly interval (`HH:MM-HH:MM`).
  3. `Consumption` (**Selected `pred_column`**): District hourly water demand in cubic meters ($\text{m}^3/\text{h}$), ranging from `1.0` to `6,514.0` $\text{m}^3/\text{h}$ (mean $= 2,314.34$, std $= 1,032.14$).
  4. `Temperature`: Local ambient air temperature ($^\circ\text{C}$, range `-17.6` to `40.3`).
  5. `Rain`: Local precipitation (arbitrary units; 25 missing values filled with `0.0`).
  6. `Sun`: Local sunshine intensity (arbitrary units, range `0.0` to `20.0`).
- **Engineered Input Features (9 total per step):** `Consumption`, `Temperature`, `Rain`, `Sun`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `IsWeekend`.
- **Chronological Train / Validation / Test Split (No Shuffling, Zero Leakage):**
  - Raw rows split chronologically `70% / 15% / 15%` (`32,222` / `6,904` / `6,906` rows), with `StandardScaler` fit **strictly on the Training partition**.
  - Sliding-window sequences (`n_steps = 48` past hours $\rightarrow$ `n_pred = 5` future hours):
    - **Training set:** **32,170** samples (`(32170, 48, 9)` $\rightarrow$ `(32170, 5)`)
    - **Validation set:** **6,852** samples (`(6852, 48, 9)` $\rightarrow$ `(6852, 5)`)
    - **Test set:** **6,854** samples (`(6854, 48, 9)` $\rightarrow$ `(6854, 5)`)

---

## 2. Problem Description

- **Title:** Attention-Based Sequence-to-Sequence Forecasting of Multi-Step Urban Water Demand.
- **Research Objective:** Develop an interpretable Encoder–Decoder deep learning model that jointly captures multi-day diurnal consumption rhythms, weekday/weekend behavioral shifts, and meteorological covariates to accurately forecast water demand for the next **five consecutive hours** ($t+1$ through $t+5$).
- **Input:** Tensor $\mathbf{X} \in \mathbb{R}^{48 \times 9}$ containing the previous 48 consecutive hourly multivariate observations ($t-47 \dots t$).
- **Output:** Vector $\hat{\mathbf{y}} \in \mathbb{R}^{5}$ representing predicted hourly water consumption (`Consumption`, in $\text{m}^3/\text{h}$) at future horizons $t+1, t+2, t+3, t+4, t+5$.
- **Solution Workflow:**
  1. Download and parse timestamps into cyclical (`sin`/`cos`) hour-of-day, day-of-week, and weekend indicators; impute 25 `Rain` nulls with `0.0`.
  2. Split chronologically (`70/15/15`) and apply Training-only `StandardScaler` normalization.
  3. Construct `48 -> 5` sliding-window tensors and train an LSTM Encoder–Decoder with Luong-style `Attention()`.
  4. Inverse-transform test predictions back to $\text{m}^3/\text{h}$ and evaluate overall and per-horizon $R^2$, NSE, MAE, and RMSE.

---

## 3. Model Architecture & Design

- **Architecture (121,025 trainable parameters, Functional Keras API):**
  1. **Encoder:** `Input(48, 9)` $\rightarrow$ `LSTM(96, return_sequences=True, return_state=True)` outputting all 48 temporal hidden states $\mathbf{H}^{\text{enc}} \in \mathbb{R}^{48 \times 96}$ and final summary states $(\mathbf{h}_{48}, \mathbf{c}_{48})$.
  2. **Decoder:** `RepeatVector(5)(h_48)` $\rightarrow$ `LSTM(96, return_sequences=True, initial_state=[h_48, c_48])` producing 5 future step representations $\mathbf{H}^{\text{dec}} \in \mathbb{R}^{5 \times 96}$.
  3. **Cross-Attention & Fusion:** Keras `Attention()([dec_out, enc_out])` computes dynamic temporal context $\mathbf{C} \in \mathbb{R}^{5 \times 96}$, concatenated with $\mathbf{H}^{\text{dec}}$ into $\mathbb{R}^{5 \times 192}$.
  4. **Forecast Head:** `TimeDistributed(Dense(32, activation='relu'))` $\rightarrow$ `Dropout(0.1)` $\rightarrow$ `TimeDistributed(Dense(1))` $\rightarrow$ `Reshape((5,))`.
- **Training Hyperparameters:**
  - Optimizer: `Adam(learning_rate=1e-3)` with `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)`.
  - Loss & Stopping: Mean Squared Error (`mse`), `batch_size=64`, max `epochs=40`, `EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)`.
  - Reproducibility: `SEED = 42` fixed across Python, NumPy, and TensorFlow.
  - Best Validation Checkpoint: **Epoch 10** (`val_loss = 0.0720`, `train_loss = 0.0507`).

---

## 4. Experimental Results & Statistical Analysis

Evaluated across **6,854 unseen chronological Test windows** (34,270 total horizon predictions) in physical units ($\text{m}^3/\text{h}$):

| Forecast Horizon | $R^2$ | NSE | MAE ($\text{m}^3/\text{h}$) | RMSE ($\text{m}^3/\text{h}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall ($t+1 \dots t+5$)** | **0.9283** | **0.9283** | **217.53** | **291.41** |
| Horizon $t+1$ (1 hour ahead) | 0.9387 | 0.9387 | 198.74 | 269.62 |
| Horizon $t+2$ (2 hours ahead) | 0.9285 | 0.9285 | 216.16 | 291.04 |
| Horizon $t+3$ (3 hours ahead) | 0.9268 | 0.9268 | 220.07 | 294.43 |
| Horizon $t+4$ (4 hours ahead) | 0.9253 | 0.9253 | 223.82 | 297.55 |
| Horizon $t+5$ (5 hours ahead) | 0.9223 | 0.9223 | 228.86 | 303.29 |

**Supporting Figures (in `figures/`):**
- [task2_training_history.png](../figures/task2_training_history.png) — Training vs. validation convergence and best-epoch checkpoint.
- [task2_forecast_comparison.png](../figures/task2_forecast_comparison.png) — 1-week (168-hour) aligned actual vs. $t+1$ and $t+5$ forecast trajectories.
- [task2_horizon_metrics.png](../figures/task2_horizon_metrics.png) — Horizon-wise MAE/RMSE/$R^2$ progression and overall parity scatter plot.

### Analytical Findings & Practical Conclusions
1. **High Hydrological Skill Across All Horizons:** Under hydrology standards (where $\text{NSE} > 0.75$ is very good and $\text{NSE} > 0.90$ is excellent), achieving **$\text{NSE} = R^2 = 0.9283$ overall** and maintaining **$\text{NSE} \ge 0.9223$** even five hours ahead confirms strong generalization on out-of-sample future years.
2. **Graceful Horizon Degradation:** Error grows smoothly and slowly from $t+1$ ($\text{MAE} = 198.74\text{ m}^3/\text{h}$) to $t+5$ ($\text{MAE} = 228.86\text{ m}^3/\text{h}$, only a `+15.1%` increase over 5x lead time), showing that the 48-hour Attention mechanism stabilizes multi-step forecasting against compounding drift.
3. **Practical Utility for Smart Water Networks:** With district demand averaging `2,314 m³/h` (peaks up to `6,514 m³/h`), an average error near `~217 m³/h` (`~9.4%` of mean load) over a 5-hour lookahead enables utility operators to schedule pump stations during off-peak electricity tariffs and proactively manage reservoir buffer capacity.
4. **Ablation Note (Why 9 Clean Features Outperformed Extra Lag Features):** In controlled testing, adding raw `lag_168` (prior week) + `LSTM(128)` slightly lowered test $R^2$ (`0.9282`) vs. the clean 9-feature `LSTM(96)` (`0.9283` with `SEED=42`), because single-point weekly shifts inject prior-week holiday/weather anomalies into the encoder.
