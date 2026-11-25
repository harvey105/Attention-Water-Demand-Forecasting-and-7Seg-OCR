# Task 3 Report Guide — Seven-Segment Display Image-to-Sequence Recognition

> **Purpose:** Complete structured analysis for **Part 2 (Task 3)** and **Part 3 (Conclusion)** of [Final Major Assignment Report 2026.docx](../Final%20Major%20Assignment%20Report%202026.docx), aligned with [Final Project Requirements 2026.pdf](../Final%20Project%20Requirements%202026.pdf).  
> **Source Code:** [task3_prepare_data.py](../task3_prepare_data.py) (73 lines) · [task3_cnn_lstm_attention.py](../task3_cnn_lstm_attention.py) (169 lines)

---

## 1. Dataset Description

- **Dataset Name:** *7SEG_OCR* (`MiXaiLL76/7SEG_OCR`, synthetic seven-segment digital display OCR dataset).
- **Download Link:** `https://huggingface.co/datasets/MiXaiLL76/7SEG_OCR` (hosted as `data/train-00000-of-00001.parquet`, ~192.5 MB).
- **Scale & Format:** **3,333 RGB images** extracted to PNG format (`data/raw/7seg_images/`) paired with string labels in `data/raw/7seg_labels.csv`.
- **Visual Content & Image Dimensions:** Multi-digit LED/LCD seven-segment displays across diverse board colors (red, green, blue, yellow, cyan, purple) and realistic unlit background segments (e.g., TM1637 modules). Original dimensions vary (e.g., `232×96` to `818×233`); all images are resized uniformly to **`64 × 256 × 3` (H × W × RGB)** and normalized to `[0, 1]`.
- **Output Sequences & Vocabulary:** Variable-length strings from **2 to 6 characters** over a **12-character vocabulary** (`-.0123456789`), expanded to **15 tokens** with `<PAD>`, `<SOS>`, and `<EOS>` (`max_dec_len = 7`).
- **Length-Stratified Train / Validation / Test Split (`80% / 10% / 10%`, `random_state=42`):**
  - **Training set:** **2,666** images
  - **Validation set:** **333** images
  - **Test set:** **334** images

---

## 2. Problem Description

- **Title:** Spatial-Aware CNN–LSTM with Multi-Head Attention for Seven-Segment Display Sequence Recognition.
- **Research Objective:** Build an end-to-end Image-to-Sequence neural model that converts a raw RGB display image directly into its ordered numeric/symbol string without manual character segmentation.
- **Input:** Normalized RGB image tensor $\mathbf{I} \in [0, 1]^{64 \times 256 \times 3}$ (plus discrete positional index vectors for encoder slices `0..63` and decoder steps `0..6`).
- **Output:** Variable-length character sequence $\hat{S} = (\hat{c}_1, \dots, \hat{c}_k)$ from `{0–9, ., -}` terminated by `<EOS>`.
- **Solution Workflow:**
  1. Download Hugging Face Parquet, extract 3,333 PNGs, and perform stratified `80/10/10` splitting by sequence length.
  2. Extract spatial visual sequences via a 4-block CNN backbone preserving `8 × 64` resolution, refine horizontal context with `Conv1D`, and inject learned positional embeddings.
  3. Decode character tokens autoregressively via an LSTM Decoder attending over the 64 horizontal visual slices with 4-head `MultiHeadAttention`.
  4. Evaluate unseen Test images via free-running greedy decoding using Exact-Match Sequence Accuracy, Character Error Rate (CER), and Character Accuracy ($1 - \text{CER}$).

---

## 3. Model Architecture & Design

- **Architecture (886,735 total parameters: 885,999 trainable, 736 BatchNorm non-trainable):**
  1. **CNN Visual Encoder (`64×256×3` $\rightarrow$ `64×128` sequence):**
     - 4 convolutional blocks `(48, 64, 128, 128)` filters (`3×3`, `padding='same'`, `relu` + `BatchNormalization`) with asymmetric pooling `(2, 2)`, `(2, 2)`, `(2, 1)`, `None`.
     - Preserving **8 vertical bands** and **64 horizontal steps** prevents tiny 2–3 pixel decimal points (`.`) from vanishing during pooling.
     - `Permute((2, 1, 3))` $\rightarrow$ `Reshape((64, 1024))` $\rightarrow$ `Dense(128, relu)` $\rightarrow$ `Conv1D(128, kernel_size=3, padding='same', relu)` + learned `Embedding(64, 128)` positional encoding.
  2. **Order-Preserving State Initialization:**
     - `MaxPooling1D(pool_size=8)` $\rightarrow$ `Flatten()` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ two `Dense(128, tanh)` heads for `(init_h, init_c)`. This compresses 64 slices into 8 left-to-right spatial regions while retaining sharp peak activations (decimal dots and minus signs) with ~8x fewer parameters than full flattening.
  3. **LSTM Decoder + Multi-Head Cross-Attention:**
     - `Embedding(15, 64)` + decoder positional `Embedding(7, 64)` $\rightarrow$ `LSTM(128, return_sequences=True)` initialized with `[init_h, init_c]`.
     - `MultiHeadAttention(num_heads=4, key_dim=32)` attends from decoder states over the 64 position-aware visual slices.
     - `Concatenate` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `TimeDistributed(Dense(64, relu))` $\rightarrow$ `TimeDistributed(Dense(15, softmax))`.
- **Training Configuration:**
  - Loss: `sparse_categorical_crossentropy` with `sample_weight` masking `<PAD>` tokens (`0.0`).
  - Optimizer & Callbacks: `Adam(1e-3)`, `ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5)`, `EarlyStopping(patience=10, restore_best_weights=True)`, `batch_size=32`, `SEED=42`.
  - Best Validation Checkpoint: **Epoch 21** (of 31 executed) with **`val_loss = 0.0546`** and **`val_accuracy = 98.01%`**.

---

## 4. Experimental Results & Iterative Optimization Analysis

### 4.1 Optimization Progression
| Model Iteration | Key Architectural Change | Exact-Match Seq Acc | CER | Char Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| Baseline v1 (`70/15/15`, `W=32`) | `Attention()` without positional encoding | `60.00%` | `14.01%` | `85.99%` |
| Iteration v2 (`70/15/15`, `W=32`) | Added Positional Embedding + `MultiHeadAttention` | `86.40%` | `3.25%` | `96.75%` |
| Iteration v3 (`80/10/10`, `W=64`) | Asymmetric width pooling (`64` steps) + `GaussianNoise` | `88.32%` | `2.69%` | `97.31%` |
| **Final Optimized (`80/10/10`, `8×64`)** | **Removed noise, preserved `H=8` vertical bands, `Conv1D`, `MaxPooling1D(8)` init** | **`91.32%`** *(305/334)* | **`1.89%`** | **`98.11%`** |

### 4.2 Performance by Sequence Length (Final Test Set, $N = 334$)
| Sequence Length | Test Count ($n$) | Exact Matches | Exact-Match Accuracy |
| :---: | :---: | :---: | :---: |
| 2 characters | 1 | 1 | **100.00%** |
| 3 characters | 7 | 7 | **100.00%** |
| 4 characters | 90 | 87 | **96.67%** |
| 5 characters | 162 | 146 | **90.12%** |
| 6 characters | 74 | 64 | **86.49%** |

### 4.3 Error Taxonomy & Practical Conclusions
- **100.00% Digit & Sign Perfection (`0 / 334` Digit/Sign Errors):** Across all 334 unseen test displays, **every single digit (`0–9`) and minus sign (`-`) was recognized with zero errors**.
- **Remaining Edge Case (`29 / 29` Errors Are Decimal-Dot Only):** All 29 remaining mismatches (`8.68%`) involve faint synthetic decimal dots (`.`) appearing on unlit background digits (e.g., `.-9586` $\rightarrow$ `-9586`).
- **Supporting Figures (in `figures/`):**
  - [task3_training_history.png](../figures/task3_training_history.png) — Masked cross-entropy loss and token accuracy convergence.
  - [task3_evaluation_summary.png](../figures/task3_evaluation_summary.png) — Accuracy by sequence length and error-type taxonomy.
  - [task3_sample_predictions.png](../figures/task3_sample_predictions.png) — Visual grid of test displays showing exact matches and dot-only edge cases.
- **Practical Relevance:** Achieving **98.11% character accuracy** and **91.32% full-sequence exact match** (`100%` on digits/signs) with a compact `<1M`-parameter model demonstrates that lightweight CNN–LSTM–Attention architectures can reliably digitize industrial seven-segment gauges and smart-device LEDs in real time without heavy Vision Transformers.
