# Group 4 — Deep Learning Final Project Task Tracker & Optimization Plan

> **Status Legend:** `[ ]` Pending · `[/]` In Progress · `[x]` Completed  
> **Scope:** Task 2 (Multi-Step Water Demand Forecasting) & Task 3 (Seven-Segment Display Image-to-Sequence Recognition) — Task 1 omitted as permitted by the instructor.  
> **Code Submission Constraint:** Clean, self-contained `.py` scripts ($\le 150\text{--}170$ lines per main script), readable flat paths (`./data/...`), concise Vietnamese `# ...` inline annotations and docstrings for oral defense (*vấn đáp*).  
> **Deferred Until User Confirmation:** `README.md`, `.gitignore`, `requirements.txt`.

---

## Phase 1: Project Structure & Data Pipeline Refactoring
- [x] **1.1 Rename & Reorganize Data Directory**
  - Renamed `Data/` $\rightarrow$ `data/` (lowercase) with `data/raw/` and `data/processed/`.
  - Deleted old demo plots (`Data/task2_results.png`, `Data/task3_results.png`).
- [x] **1.2 Create `figures/` and `docs/` Directories**
- [x] **1.3 Remove Inline Plotting Blocks & Decouple Visualization**
  - Removed inline `matplotlib` blocks from Task 2 & Task 3 training scripts; evaluation histories & predictions export to `data/processed/task2_eval.npz` and `data/processed/task3_eval.npz`.

---

## Phase 2: Model Optimization (Target: >90% and ideally ~93–95%+ Stable)
- [x] **2.1 Error & Residual Diagnosis of Current Baseline Models**
  - Diagnosed 26/30 sampled Task 3 errors as dropped decimal dots `.` caused by 8x horizontal pooling (`W=32` slices).
- [x] **2.2 Optimize Task 2 (`task2_prepare_data.py` & `task2_seq2seq_attention.py`)**
  - Benchmarked `LSTM-96` vs. `LSTM-128` vs. `24h Residual`; selected clean 9-feature `LSTM(96)` (`121,025` params, `SEED=42`), achieving **`R² = NSE = 0.9283`** overall ($t+1$: `0.9387` $\rightarrow$ $t+5$: `0.9223`).
- [x] **2.3 Optimize Task 3 (`task3_prepare_data.py` & `task3_cnn_lstm_attention.py`)**
  - Stratified `80/10/10` split (`2,666 / 333 / 334`), preserved `8×64` spatial resolution, added `Conv1D` + `MaxPooling1D(8)` ordered state initialization, pushing **Exact-Match Seq Accuracy** from `60.00%` $\rightarrow$ **`91.32%`** (`305/334`), **CER = `1.89%`**, **Char Accuracy = `98.11%`**, with **0/334 digit/sign errors**.
- [x] **2.4 Reproducibility & Consistency Verification**
  - Deterministic `SEED = 42` set across both tasks; verified end-to-end execution from `./data/raw/` and `./data/processed/` paths.

---

## Phase 3: Code Polish & Vietnamese Oral-Defense Annotations
- [x] **3.1 Refactor & Annotate All 4 Main Scripts (`<= 150-170` lines each)**
  - `task2_prepare_data.py` (96 lines), `task2_seq2seq_attention.py` (122 lines), `task3_prepare_data.py` (73 lines), `task3_cnn_lstm_attention.py` (169 lines).
- [x] **3.2 Verify Line Counts & Code Simplicity**

---

## Phase 4: High-Resolution Figures (`generate_figures.py` $\rightarrow$ `figures/`)
- [x] **4.1 Build Standalone Plotting Script (`generate_figures.py`)**
- [x] **4.2 Execute `generate_figures.py` & Visually Inspect All `.png` / `.svg` Outputs**
  - **Task 2:** `task2_training_history`, `task2_forecast_comparison` (aligned $t+5$), `task2_horizon_metrics` (`.png` & `.svg`).
  - **Task 3:** `task3_training_history`, `task3_evaluation_summary`, `task3_sample_predictions` (`.png` & `.svg`).

---

## Phase 5: Comprehensive Report Documentation (`docs/`)
- [x] **5.1 Deep Re-Check of Requirements & Report Template**
- [x] **5.2 Write `docs/` Report-Support Markdown Files (in Academic English)**
  - `docs/00_theoretical_background.md`, `docs/01_task2_water_demand_analysis.md`, `docs/02_task3_7seg_ocr_analysis.md`, `docs/03_oral_defense_qa_prep.md`.

---

## Phase 6: Final Verification & Handoff (Pre-Confirmation)
- [x] Verify clean directory layout (`data/raw`, `data/processed`, `figures/`, `docs/`, 4 main `.py` files + `generate_figures.py`).
- [x] Confirm all scripts run end-to-end cleanly with final metrics and figures updated.
- [x] Added detailed accented Vietnamese annotations to all 4 scripts, rewrote `docs/03_oral_defense_qa_prep.md` in Vietnamese per file, generated architecture & repository diagrams, and created `README.md`, `requirements.txt`, and `.gitignore`.
