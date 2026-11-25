# Theoretical Background & Evaluation Methodology

> **Purpose:** Reference material for **Part 1: Theoretical Background** and metric definitions in **Part 2** of [Final Major Assignment Report 2026.docx](../Final%20Major%20Assignment%20Report%202026.docx), aligned with [Final Project Requirements 2026.pdf](../Final%20Project%20Requirements%202026.pdf).

---

## 1. Recurrent Neural Networks (RNNs) & Long Short-Term Memory (LSTM)

### 1.1 Standard Recurrent Neural Networks (SimpleRNN)
Recurrent Neural Networks process sequential inputs $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$ step by step while maintaining a hidden state $\mathbf{h}_t \in \mathbb{R}^d$:
$$\mathbf{h}_t = \tanh(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h)$$
Although SimpleRNN captures short-term temporal dependencies, backpropagation through time (BPTT) causes gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_k}$ to repeatedly multiply by $\mathbf{W}_{hh}^\top$, resulting in **vanishing or exploding gradients** when dependencies span dozens of time steps (e.g., 24-hour or 48-hour water demand cycles).

### 1.2 Long Short-Term Memory (LSTM)
LSTM (Hochreiter & Schmidhuber, 1997) solves the vanishing-gradient bottleneck by introducing an explicit **cell state** $\mathbf{c}_t$ regulated by three multiplicative gates:
- **Forget gate** $\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$: decides what historical information to discard from $\mathbf{c}_{t-1}$.
- **Input gate** $\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$ and candidate state $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$: decide what new information to store.
- **Cell state update** $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$: enables nearly linear gradient flow across long horizons.
- **Output gate** $\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$ and hidden state $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$.

In our project, LSTM is used both as the **temporal encoder–decoder** in Task 2 (capturing multi-day diurnal/weekly water consumption patterns) and as the **autoregressive sequence decoder** in Task 3 (generating variable-length 7-segment character tokens).

---

## 2. Encoder–Decoder (Sequence-to-Sequence) Architecture

The Encoder–Decoder (Seq2Seq) framework maps an input sequence (or spatial feature map) of length $T_{\text{in}}$ to an output sequence of length $T_{\text{out}}$:
1. **Encoder:** Compresses the input representation into a sequence of hidden states $\mathbf{H} = (\mathbf{h}_1^{\text{enc}}, \dots, \mathbf{h}_{T_{\text{in}}}^{\text{enc}})$ and final summary state $(\mathbf{h}_{T_{\text{in}}}^{\text{enc}}, \mathbf{c}_{T_{\text{in}}}^{\text{enc}})$.
   - In **Task 2**, the encoder is an LSTM ingesting $T_{\text{in}} = 48$ past hourly multivariate observations.
   - In **Task 3**, the encoder is a **Convolutional Neural Network (CNN)** that converts a 2D image $(H \times W \times 3)$ into a left-to-right sequence of $T_{\text{in}}$ spatial visual feature vectors, enriched with **Positional Embeddings** so spatial order is preserved.
2. **Decoder:** Initialized from the encoder summary state, it generates the target sequence step by step:
   - In **Task 2**, it predicts $T_{\text{out}} = 5$ consecutive future hourly water consumption values ($t+1$ through $t+5$).
   - In **Task 3**, it uses **Teacher Forcing** during training (feeding `<SOS>` + ground-truth characters) and **autoregressive greedy decoding** at inference until `<EOS>` is emitted.

---

## 3. Attention Mechanisms

A fixed-length context vector bottlenecks Seq2Seq models when input sequences are long or contain localized salient cues. **Attention** allows the decoder at step $t$ to dynamically weight every encoder position $s \in \{1, \dots, T_{\text{in}}\}$:

### 3.1 Dot-Product / Luong Attention (Task 2)
Given decoder state (query) $\mathbf{q}_t = \mathbf{h}_t^{\text{dec}}$ and encoder outputs (keys/values) $\mathbf{k}_s = \mathbf{v}_s = \mathbf{h}_s^{\text{enc}}$:
$$\alpha_{t,s} = \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_s)}{\sum_{j=1}^{T_{\text{in}}} \exp(\mathbf{q}_t^\top \mathbf{k}_j)}, \qquad \mathbf{ctx}_t = \sum_{s=1}^{T_{\text{in}}} \alpha_{t,s} \mathbf{v}_s$$
The context vector $\mathbf{ctx}_t$ is concatenated with $\mathbf{h}_t^{\text{dec}}$ so each future step ($t+1 \dots t+5$) can selectively focus on relevant past hours (e.g., same hour 24h or 48h ago).

### 3.2 Multi-Head Cross-Attention with Positional Encoding (Task 3)
Because 7-segment display images contain ordered digits from left to right (and sometimes repeated digits or small decimal dots), pure content attention without position awareness can confuse digit order (e.g., `7156` vs. `7515`). Adding learned **Positional Embeddings** to both encoder visual slices and decoder steps, and applying **Multi-Head Attention** ($h = 4$ heads):
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}, \qquad \text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$
allows different attention heads to simultaneously track spatial position, digit strokes, and decimal/minus symbols.

---

## 4. Evaluation Metrics & Sequence-to-Sequence Formulation

### 4.1 Multi-Step Time Series Metrics (Task 2)
For a multi-step test set of $N$ windows and $H = 5$ future horizons, let $y_{i,h}$ and $\hat{y}_{i,h}$ denote the actual and predicted values (in $\text{m}^3/\text{h}$) for sample $i$ at horizon $h \in \{1,\dots,5\}$. Metrics are evaluated both **per-horizon** (over $N$ samples at a fixed $h$) and **overall** (flattening all $N \times H$ predictions):

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{M}\sum_{k=1}^{M} |y_k - \hat{y}_k|$$
   Measures average absolute forecast error in physical units ($\text{m}^3/\text{h}$), robust to occasional outliers.
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{M}\sum_{k=1}^{M} (y_k - \hat{y}_k)^2}$$
   Penalizes large peak-hour errors more heavily than MAE; always $\text{RMSE} \ge \text{MAE}$.
3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{k=1}^{M} (y_k - \hat{y}_k)^2}{\sum_{k=1}^{M} (y_k - \bar{y})^2}$$
   Measures the proportion of target variance explained by the model ($1.0$ = perfect; $0.0$ = mean baseline).
4. **Nash–Sutcliffe Efficiency (NSE):**
   $$\text{NSE} = 1 - \frac{\sum_{k=1}^{M} (y_k - \hat{y}_k)^2}{\sum_{k=1}^{M} (y_k - \bar{y})^2}$$
   Originally introduced in hydrology/water-resources engineering (Nash & Sutcliffe, 1970). Note the key theoretical nuance: when $R^2$ is defined as the **regression coefficient of determination** ($1 - \text{SS}_{\text{res}}/\text{SS}_{\text{tot}}$, as implemented in `sklearn.metrics.r2_score`), $\text{NSE}$ and $R^2$ are **mathematically identical**, whereas Pearson's squared correlation $r^2$ ignores systematic bias/scale error ($\text{NSE} \le r^2$). Thresholds in water modeling typically treat $\text{NSE} > 0.75$ as *very good* and $\text{NSE} > 0.90$ as *excellent*.

### 4.2 Image-to-Sequence OCR Metrics (Task 3)
*(Note: While the assignment PDF accidentally repeats the regression metrics paragraph under Task 3, the official report template Section 3.4 specifies "appropriate evaluation metrics" for image-to-sequence prediction.)*
1. **Exact-Match Sequence Accuracy:**
   $$\text{Exact-Match Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}(\hat{S}_i = S_i)$$
   Strictest metric: a prediction is counted correct only if **every** character, decimal point, and sign matches the ground truth exactly.
2. **Character Error Rate (CER) & Character Accuracy:**
   $$\text{CER} = \frac{\sum_{i=1}^{N} \text{Levenshtein}(\hat{S}_i, S_i)}{\sum_{i=1}^{N} |S_i|}, \qquad \text{Char Acc} = 1 - \text{CER}$$
   Quantifies the normalized edit distance (insertions, deletions, substitutions) across all ground-truth characters.
