# Tài Liệu Ôn Tập Vấn Đáp (Oral Defense Q&A) — Giải Thích Sâu Từng Dòng & Câu Hỏi Hỏi Vặn

> **Mục đích:** Tài liệu chuẩn bị vấn đáp cuối kỳ bằng **tiếng Việt** cho cả 4 thành viên Nhóm 4.  
> **Cách dùng khi thi:** Giảng viên thường hỏi ngẫu nhiên mỗi sinh viên **1 file code riêng biệt** và hay hỏi vặn (*"tại sao lại dùng hàm này / tại sao đặt số này / nếu đổi sang X thì sao?"*). Vì vậy phần 2 tách riêng từng file code, giải thích **từ trên xuống dưới** kèm các câu hỏi vặn điển hình.

---

## 1. General Pipeline & Reproducibility Questions (Câu Hỏi Chung Toàn Dự Án)

1. **Tại sao nhóm lại tách `data/raw/` và `data/processed/`, đồng thời tách file vẽ biểu đồ `generate_figures.py` riêng?**
   - `data/raw/` chỉ chứa dữ liệu thô tải từ Zenodo/Hugging Face (không bị ghi đè hay sửa hỏng). `data/processed/` chứa các tensor `.npz`, trọng số đã train (`.keras`) và kết quả test (`task2_eval.npz`, `task3_eval.npz`).
   - Tách riêng `generate_figures.py` giúp 4 file code chính cực gọn (`78–172` dòng), chỉ làm đúng 2 việc: chuẩn bị dữ liệu và train/đánh giá mô hình. Khi muốn chỉnh màu/tiêu đề biểu đồ báo cáo chỉ cần chạy lại `generate_figures.py` (~10 giây), không phải train lại mô hình mất hàng chục phút.

2. **Tại sao phải đặt `SEED = 42` ở tận 4 chỗ (`PYTHONHASHSEED`, `random`, `np.random`, `tf.random`)?**
   - Quá trình train có nhiều bước ngẫu nhiên: khởi tạo trọng số mạng nơ-ron (`kernel_initializer`), xáo trộn mini-batch mỗi epoch, và tắt ngẫu nhiên neuron trong `Dropout`. Nếu chỉ đặt `np.random.seed(42)` mà quên `tf.random.set_seed(42)` thì mỗi lần chạy lại TensorFlow sẽ ra metrics hơi khác nhau. Đặt đồng bộ cả 4 chỗ giúp đảm bảo **tính tái lập (reproducibility)**. *(Con số `42` là quy ước phổ biến trong khoa học dữ liệu, có thể đổi sang bất kỳ số nguyên nào miễn là giữ cố định).*

3. **Tại sao ép kiểu tất cả mảng về `float32` thay vì giữ mặc định `float64` của NumPy?**
   - Trọng số (weights) của các lớp trong TensorFlow/Keras mặc định là `float32`. Nếu đầu vào là `float64`, Keras phải tự động ép kiểu ngầm gây tốn thời gian và làm **tốn gấp đôi RAM** (đặc biệt với 32,170 cửa sổ 48×9 ở Task 2 và 2,666 ảnh 64×256×3 ở Task 3). `float32` giữ đủ độ chính xác (~7 chữ số có nghĩa) mà nhẹ gấp đôi.

4. **Tại sao cả 2 Task đều chia 3 tập `Train / Validation / Test` thay vì chia 2 tập `Train / Test` (gán `X_val = X_test` như code mẫu)?**
   - Trong 2 file demo ngắn của giảng viên, `X_val = X_test` chỉ dùng để minh họa nhanh. Nhưng theo chuẩn báo cáo đồ án (mục 1.1 & 3.1 của file `.docx`) và chuẩn khoa học dữ liệu, nếu dùng tập Test làm Validation để `EarlyStopping` chọn epoch tốt nhất thì tức là mô hình đã **gián tiếp nhìn thấy tập Test** (data leakage). Tách 3 tập riêng đảm bảo **Test là dữ liệu hoàn toàn chưa từng tham gia vào quá trình chọn mô hình**.

5. **Trong file PDF yêu cầu ghi Task 3 đánh giá bằng `R², NSE, MAE, RMSE`, tại sao nhóm lại dùng `Exact-Match Accuracy` và `CER`?**
   - Đây là lỗi copy-paste nhẹ trong đề PDF (đoạn cuối trang 2 lặp lại y nguyên đoạn metrics của Task 2). Trong file mẫu báo cáo `.docx` mục 3.4 ghi rõ *"appropriate evaluation metrics"*, và bài toán Task 3 là **Image-to-Sequence OCR** (đầu ra là chuỗi ký tự rời rạc `-.0123456789` có độ dài thay đổi 2–6 ký tự, không phải bài toán hồi quy liên tục). Vì vậy dùng **Exact-Match Sequence Accuracy** (đoán đúng 100% cả chuỗi) và **CER** (tỷ lệ lỗi ký tự theo khoảng cách Levenshtein) mới đúng bản chất bài toán.

---

## 2. Giải Thích Chi Tiết Từng File Code (Từ Trên Xuống Dưới & Câu Hỏi Hỏi Vặn)

### 2.1. File `task2_prepare_data.py` (Chuẩn Bị Dữ Liệu Task 2 — 98 dòng)

#### A. Giải thích từ trên xuống dưới
1. **Imports & Hằng số:**
   - `os`: tạo thư mục (`makedirs`) và nối đường dẫn (`os.path.join`) chuẩn trên cả Windows (`\`) và Linux/Colab (`/`).
   - `requests`: gửi yêu cầu HTTP GET tải file CSV trực tiếp từ link Zenodo.
   - `numpy`, `pandas`: đọc bảng CSV (`46,032` dòng) và tính toán ma trận, hàm `sin`/`cos`.
   - `StandardScaler` (`sklearn.preprocessing`): chuẩn hóa dữ liệu về trung bình `0`, độ lệch chuẩn `1` theo công thức $z = \frac{x - \mu}{\sigma}$.
2. **Hàm `download_dataset()`:**
   - `os.makedirs(RAW_DIR, exist_ok=True)`: `exist_ok=True` nghĩa là *"nếu thư mục đã tồn tại rồi thì bỏ qua, không báo lỗi `FileExistsError`"*.
   - `if not os.path.exists(CSV_PATH)`: kiểm tra xem máy đã tải file chưa; nếu có rồi thì dùng lại luôn để tiết kiệm thời gian.
   - `requests.get(ZENODO_URL, timeout=60)`: đặt giới hạn chờ 60 giây để chương trình không bị treo vĩnh viễn nếu mạng đứt; `resp.raise_for_status()` lập tức báo lỗi nếu server trả mã lỗi (404, 500...); mở file chế độ `'wb'` (*write binary*) để ghi nguyên vẹn byte tải về.
3. **Hàm `create_sequences(values, target_col, n_steps, n_pred)`:**
   - Vòng lặp `for i in range(len(values) - (n_steps + n_pred) + 1)`:
     - Tại mỗi vị trí `i`, đầu vào `X` lấy `n_steps = 48` giờ quá khứ `[i : i + 48, :]` (dấu `:` lấy toàn bộ `9` cột đặc trưng).
     - Đầu ra `y` lấy `n_pred = 5` giờ tương lai liên tiếp `[i + 48 : i + 53, target_col]` (chỉ lấy cột `target_col = 0` là `Consumption`).
     - Phải trừ `(n_steps + n_pred)` và `+ 1` để vòng lặp dừng đúng tại vị trí cuối cùng còn đủ 48 + 5 = 53 dòng, tránh lỗi vượt quá độ dài mảng.
4. **Hàm `main()`:**
   - **Tách thời gian:** `pd.to_datetime(..., format='%m/%d/%Y')` ép đúng định dạng Mỹ `tháng/ngày/năm` (nếu không ghi rõ `format`, Pandas có thể nhầm ngày `4/1/2021` thành 4 tháng 1 hoặc 1 tháng 4). Cột `Time` dạng `" 00:00-01:00 "` được `.str.strip()` xóa khoảng trắng thừa rồi `.str.slice(0, 2)` lấy 2 ký tự giờ đầu (`00` $\rightarrow$ `0`).
   - **`IsWeekend = (DayOfWeek >= 5)`:** Vì `day_of_week` đánh số Thứ 2 = `0` .. Thứ 6 = `4`, Thứ 7 = `5`, Chủ nhật = `6`, nên `>= 5` tách đúng 2 ngày cuối tuần (hành vi dùng nước của hộ gia đình/nhà máy khác hẳn ngày thường).
   - **Mã hóa vòng tròn (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`):**
     - Công thức: $\sin(2\pi \cdot \text{Hour}/24)$ và $\cos(2\pi \cdot \text{Hour}/24)$.
     - *Hỏi vặn: Tại sao không giữ nguyên `Hour` là số nguyên `0..23`?* $\rightarrow$ Nếu để số nguyên, khoảng cách giữa `23:00` và `00:00` bị hiểu là $|23 - 0| = 23$ (rất xa nhau), dù thực tế chúng chỉ cách nhau 1 giờ! Chiếu lên đường tròn lượng giác bằng cặp `(sin, cos)` giúp `23h` và `0h` nằm sát cạnh nhau liên tục.
   - **`df['Rain'].fillna(0.0)`:** Bộ dữ liệu đã làm sạch gần hết nhưng cột `Rain` vẫn còn đúng `25` giá trị `NaN`. Điền `0.0` (không mưa) giúp không bị mất dòng thời gian liên tục.
   - **Chia Train/Val/Test (`70% / 15% / 15%`) theo trình tự thời gian:** Cắt thẳng bằng chỉ số mảng (`values[:n_train]`, `values[n_train:n_train+n_val]`, `values[n_train+n_val:]`), **tuyệt đối không dùng `shuffle=True`**.
   - **Fit `StandardScaler` chỉ trên `train_raw`:** Tính $\mu, \sigma$ duy nhất từ 70% dữ liệu đầu (quá khứ), sau đó dùng chính $\mu, \sigma$ đó để `.transform()` cho Val và Test. Cuối cùng lưu `target_mean` và `target_std` vào `.npz` để file train biết cách nhân/cộng ngược về đơn vị thực $\text{m}^3/\text{h}$.

#### B. Câu hỏi "hỏi vặn" thường gặp cho `task2_prepare_data.py`
- **Hỏi:** *Tại sao `n_steps = 48` mà không để `n_steps = 3` như file mẫu của thầy?*
  - **Trả lời:** Nước sinh hoạt có chu kỳ 24 giờ rất mạnh (hệ số tự tương quan lag-24h = `0.9123`, trong khi lag-5h tụt xuống `0.0338` do lệch pha ngày/đêm). Nếu chỉ nhìn 3 giờ trước, mô hình không biết giờ này hôm qua tiêu thụ bao nhiêu; dùng `48` giờ (2 chu kỳ ngày-đêm) giúp Encoder quan sát trọn vẹn đỉnh sáng/tối của cả 2 ngày gần nhất.
- **Hỏi:** *Nếu gọi `scaler.fit_transform(values)` trên toàn bộ bảng trước khi chia Train/Test thì sao?*
  - **Trả lời:** Sẽ bị **Data Leakage (rò rỉ dữ liệu tương lai)** vì thông tin trung bình và độ lệch chuẩn của tập Test đã bị trộn vào bước chuẩn hóa của tập Train, làm kết quả đánh giá bị ảo cao hơn thực tế.
- **Hỏi:** *Tại sao file mẫu của thầy không chuẩn hóa `StandardScaler` mà nhóm lại dùng?*
  - **Trả lời:** Vì cột `Consumption` dao động tới `6,514`, trong khi `hour_sin/cos` và `IsWeekend` nằm trong `[-1, 1]`. Không chuẩn hóa sẽ khiến trọng số đầu vào của `Consumption` lấn át hoàn toàn các biến thời tiết/thời gian và làm hàm kích hoạt của LSTM dễ bão hòa.

---

### 2.2. File `task2_seq2seq_attention.py` (Huấn Luyện & Đánh Giá Task 2 — 127 dòng)

#### A. Giải thích từ trên xuống dưới
1. **Imports & `set_seed()`:**
   - Dùng **Keras Functional API** (`Input`, `Model`) thay vì `Sequential` vì `Sequential` chỉ nối thẳng một mạch, không thể truyền `initial_state=[state_h, state_c]` từ Encoder sang Decoder hay rẽ nhánh nối tầng `Attention`.
2. **Hàm `nse_score(y_true, y_pred)`:**
   - Công thức: $\text{NSE} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.
   - Ý nghĩa: So sánh sai số bình phương của mô hình (tử số) với sai số nếu chỉ đoán bằng giá trị trung bình $\bar{y}$ (mẫu số). $\text{NSE} = 1$ là hoàn hảo, $\text{NSE} = 0$ nghĩa là mô hình chỉ tốt ngang đoán trung bình, $\text{NSE} > 0.90$ được xếp loại xuất sắc.
3. **Hàm `evaluate_forecasts(y_true, y_pred)`:**
   - `y_true.reshape(-1)`: duỗi ma trận `(6854, 5)` thành vector 1 chiều `34,270` phần tử để tính **Overall** 4 chỉ số $R^2$, NSE, MAE, RMSE trên toàn bộ 5 bước dự báo.
   - Vòng `for step in range(y_true.shape[1])`: tách riêng từng cột `step = 0..4` (tương ứng $t+1 \dots t+5$) để kiểm chứng mô hình có bị suy giảm mạnh ở bước xa $t+5$ không.
4. **Hàm `build_seq2seq_attention(n_steps, n_features, n_pred)`:**
   - **Encoder `LSTM(96, return_sequences=True, return_state=True)`:**
     - `96` units: đủ lớn để học quan hệ phi tuyến 48 giờ, nhưng gọn hơn `128` (nhóm đã thử nghiệm `128` bị overfit nhẹ nên `R² = 0.9282` với nhiều tham số hơn, còn `96` đạt `0.9283` ổn định nhất với `121,025` tham số).
     - `return_sequences=True`: trả ra cả chuỗi 48 trạng thái ẩn `(batch, 48, 96)` làm **Keys/Values** cho tầng Attention.
     - `return_state=True`: trả thêm `(state_h, state_c)` ở bước thứ 48 làm trạng thái khởi tạo cho Decoder.
   - **Decoder `RepeatVector(n_pred)` + `LSTM(96, return_sequences=True)`:**
     - Vì Decoder phải sinh ra `5` bước tương lai nhưng chưa có chuỗi đầu vào tương lai, `RepeatVector(5)(state_h)` sao chép vector ngữ cảnh `state_h` thành `(batch, 5, 96)` để kích hoạt 5 bước lặp của Decoder LSTM (khởi tạo bằng `initial_state=[state_h, state_c]`).
   - **Attention & Đầu ra:**
     - `Attention()([dec_out, enc_out])`: triển khai Luong dot-product attention — từng bước tương lai trong `dec_out` (Query) tự tính trọng số chú ý lên cả 48 giờ quá khứ trong `enc_out` (Key/Value) để tạo vector `context` `(batch, 5, 96)`.
     - `Concatenate(axis=-1)([dec_out, context])`: ghép đôi thông tin Decoder và ngữ cảnh Attention trên trục cuối cùng (`96 + 96 = 192` chiều).
     - `TimeDistributed(Dense(32, activation='relu'))` + `Dropout(0.1)` + `TimeDistributed(Dense(1))`: dùng chung một bộ trọng số MLP cho cả 5 bước dự báo, cuối cùng `Reshape((5,))` bỏ trục `1` ở cuối cho khớp shape của `y_train`.
5. **Hàm `main()`:**
   - `EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)`: nếu 7 epoch liên tiếp `val_loss` không giảm thì dừng và **bắt buộc bật `restore_best_weights=True`** để quay về trọng số tại epoch tốt nhất (Epoch 10, `val_loss = 0.0720`) thay vì giữ epoch cuối (Epoch 17).
   - `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)`: nếu 3 epoch `val_loss` không cải thiện, tự động chia đôi tốc độ học giúp tìm cực tiểu chính xác hơn.
   - Quy đổi ngược: `y_pred = model.predict(...) * target_std + target_mean` trước khi tính MAE/RMSE bằng đơn vị thực $\text{m}^3/\text{h}$.

#### B. Câu hỏi "hỏi vặn" thường gặp cho `task2_seq2seq_attention.py`
- **Hỏi:** *Tại sao trên kết quả in ra, giá trị $R^2$ và NSE lại bằng nhau y hệt (`0.9283`)? Có tính nhầm không?*
  - **Trả lời:** Không nhầm! Theo định nghĩa của `sklearn.metrics.r2_score`, hệ số xác định là $R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$, trùng khớp hoàn toàn về công thức toán học với **NSE** trong thủy văn. Chỉ khi tính theo bình phương hệ số tương quan Pearson ($r^2$) thì mới khác NSE (vì Pearson $r^2$ bỏ qua sai lệch hệ số chặn/độ lệch tỷ lệ).
- **Hỏi:** *Tại sao không dùng `SimpleRNN` như file mẫu mà lại dùng `LSTM`?*
  - **Trả lời:** Cửa sổ đầu vào của nhóm dài `48` bước (so với `3` bước trong file mẫu). `SimpleRNN` bị suy giảm gradient (*vanishing gradient*) khi truyền ngược qua 48 bước, không nhớ được cùng giờ ngày hôm qua (`t-24`). `LSTM` có `cell state` và các cổng quên/nhớ giúp giữ thông tin dài hạn cực tốt.
- **Hỏi:** *Nếu tắt `return_sequences=True` ở Encoder thì code báo lỗi gì?*
  - **Trả lời:** Encoder chỉ trả về tensor 2D `(batch, 96)` thay vì 3D `(batch, 48, 96)`, khi đưa vào `Attention()([dec_out, enc_out])` sẽ lỗi kích thước và mất toàn bộ 48 điểm thời gian để chú ý.

---

### 2.3. File `task3_prepare_data.py` (Chuẩn Bị Dữ Liệu Ảnh 7SEG_OCR — 78 dòng)

#### A. Giải thích từ trên xuống dưới
1. **Imports & Hằng số:**
   - `io`: dùng `io.BytesIO` bọc chuỗi byte ảnh nằm trong cột Parquet thành luồng dữ liệu giống file trong RAM để PIL mở trực tiếp.
   - `PIL.Image`: mở ảnh và chuyển đổi hệ màu.
   - `train_test_split`: chia tập dữ liệu phân tầng (`stratify`).
2. **Hàm `download_parquet()`:**
   - `requests.get(HF_PARQUET_URL, stream=True)`: đặt `stream=True` kết hợp `r.iter_content(chunk_size=1024*1024)` để tải và ghi dần từng `1 MB` ra đĩa, tránh nạp cả ~192 MB vào RAM cùng lúc gây tràn bộ nhớ ở máy yếu.
3. **Hàm `main()`:**
   - `pd.read_parquet(PARQUET_PATH)`: đọc bảng dữ liệu `3,333` ảnh từ Hugging Face.
   - Vòng `for idx, row in df.iterrows()`:
     - `img_obj['bytes'] if isinstance(img_obj, dict) else img_obj`: phòng trường hợp các phiên bản Pandas/PyArrow khác nhau đọc cột ảnh thành `dict` hoặc `bytes` thuần.
     - `filename = f'img_{idx:05d}.png'`: định dạng 5 chữ số có số 0 ở đầu (`00000` $\rightarrow$ `03332`) để sắp xếp theo tên file luôn đúng thứ tự.
     - `.convert('RGB')`: ép tất cả ảnh về chuẩn 3 kênh màu (Red-Green-Blue), tránh lỗi crash nếu có ảnh RGBA (4 kênh trong suốt) hoặc ảnh xám (1 kênh).
   - **Chia 3 tập `80% / 10% / 10%`:**
     - Tạo cột tạm `seq_len = meta_df['label'].str.len()` (độ dài chuỗi từ `2` đến `6` ký tự).
     - Lần 1: `train_test_split(meta_df, test_size=0.20, random_state=42, stratify=meta_df['seq_len'])` $\rightarrow$ tách `2,666` ảnh Train (80%) và `667` ảnh `temp_df` (20%).
     - Lần 2: `train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['seq_len'])` $\rightarrow$ chia đôi `20%` thành `333` ảnh Val (10%) và `334` ảnh Test (10%).
     - Lưu ra `7seg_labels.csv` với `index=False` (không ghi cột index thừa).

#### B. Câu hỏi "hỏi vặn" thường gặp cho `task3_prepare_data.py`
- **Hỏi:** *Tại sao phải chia 2 lần với `test_size=0.20` rồi `0.50`? Tại sao dùng `stratify=meta_df['seq_len']`?*
  - **Trả lời:** `train_test_split` chỉ chia 1 lần thành 2 phần, nên để chia 3 phần `80/10/10` phải tách `20%` ra trước rồi chia đôi (`0.50 * 0.20 = 0.10`). Dùng `stratify` theo độ dài chuỗi đảm bảo tỷ lệ ảnh dài `2, 3, 4, 5, 6` ký tự phân bố đều như nhau trong Train/Val/Test (ví dụ không bị tình trạng toàn bộ chuỗi 6 ký tự rơi hết vào tập Test).
- **Hỏi:** *Tại sao không chuyển ảnh sang trắng-đen (Grayscale 1 kênh) cho nhẹ hơn?*
  - **Trả lời:** Ảnh trong `7SEG_OCR` là bảng LED nhiều màu (chữ đỏ trên nền đỏ sẫm, chữ xanh lá trên viền xanh, chữ vàng trên nền tím...). Nếu chuyển sang ảnh xám, độ tương phản giữa nét LED sáng và nền bo mạch bị giảm đi ở một số bảng màu, khiến dấu chấm thập phân (`.`) rất mờ dễ mất hẳn. Giữ 3 kênh RGB giúp CNN phân biệt màu nét sáng tốt hơn.

---

### 2.4. File `task3_cnn_lstm_attention.py` (Huấn Luyện & Đánh Giá Task 3 — 172 dòng)

#### A. Giải thích từ trên xuống dưới
1. **Hằng số đầu file:**
   - `IMG_H = 64, IMG_W = 256, ENC_STEPS = 64`: Resize ảnh về khung ngang `64×256` khớp tỷ lệ màn hình LED nhiều chữ số. Quan trọng: giữ **`ENC_STEPS = 64`** (`256 / 4`) thay vì pool xuống `32` để các nét mảnh như dấu `.` và `-` không bị mất.
   - `PAD_TOKEN = '<PAD>'`, `SOS_TOKEN = '<SOS>'`, `EOS_TOKEN = '<EOS>'`: `3` token điều khiển giúp mô hình xử lý chuỗi có độ dài thay đổi (2–6 ký tự) trong cùng một tensor cố định.
2. **Hàm `levenshtein_distance(s1, s2)`:**
   - Thuật toán quy hoạch động đếm số phép sửa tối thiểu (**chèn**, **xóa**, **thay thế** 1 ký tự) để biến chuỗi `s1` thành `s2`.
   - Trong vòng lặp `for i`, `prev[j + 1] + 1` là phép xóa, `curr[j] + 1` là phép chèn, `prev[j] + (c1 != c2)` là phép thay thế (cộng 1 nếu 2 ký tự khác nhau, cộng 0 nếu giống nhau). Kết quả cuối cùng `prev[-1]` dùng làm tử số tính chỉ số **CER**.
3. **Hàm `load_split_data(df, char_to_id, max_dec_len)`:**
   - Tạo sẵn `dec_in` và `dec_out` kích thước `(N, 7)` điền đầy bằng ID của `<PAD>`.
   - `/ 255.0`: chuẩn hóa giá trị điểm ảnh từ `[0, 255]` về `[0, 1]` giúp gradient của Conv2D ổn định.
   - **Cơ chế Teacher Forcing:** Chuỗi đầy đủ là `seq = [<SOS>, c1, ..., cn, <EOS>]`. Ta gán `dec_in = seq[:-1]` (`[<SOS>, c1, ..., cn]`) và `dec_out = seq[1:]` (`[c1, ..., cn, <EOS>]`). Nghĩa là khi train, ở bước $t$, Decoder nhận **ký tự đúng của bước $t-1$** để dự đoán ký tự ở bước $t$.
   - `sample_weights = (dec_out != PAD_ID).astype('float32')`: Tạo mặt nạ bằng `1.0` ở vị trí ký tự thật và `0.0` ở vị trí `<PAD>` để Keras không tính loss hay cộng điểm accuracy khống cho các ô đệm `<PAD>`.
4. **Hàm `build_cnn_lstm_attention(vocab_size, max_dec_len)`:**
   - **CNN Encoder (4 khối):** Vòng `for` lặp qua `[(48, (2,2)), (64, (2,2)), (128, (2,1)), (128, None)]`:
     - Mỗi khối gồm `Conv2D(3×3, padding='same', activation='relu')` $\rightarrow$ `BatchNormalization()` $\rightarrow$ `MaxPooling2D(pool)`.
     - Chú ý pooling bất đối xứng: chiều cao giảm `2×2×2 = 8` lần (`64 -> 8`), chiều rộng chỉ giảm `2×2 = 4` lần (`256 -> 64`), và khối thứ 4 **không pooling (`None`)** để giữ nguyên độ phân giải `8×64`, bảo toàn các dấu chấm `.` nhỏ ở chân chữ số.
   - **Chuyển ảnh 2D thành chuỗi 1D trái $\rightarrow$ phải:**
     - `Permute((2, 1, 3))` đổi trục `(H=8, W=64, C=128)` thành `(W=64, H=8, C=128)`, rồi `Reshape((64, 1024))` gộp chiều cao và kênh màu lại thành **64 lát cắt theo chiều ngang**.
     - `Dense(128)` giảm từ `1024` xuống `128` chiều; `Conv1D(128, kernel_size=3, padding='same')` giúp mỗi lát cắt kết hợp ngữ cảnh với 2 lát cắt kề trái/phải.
   - **Positional Embedding (`Embedding(64, 128)` + `Add()`):** Cộng vector mã hóa vị trí `0..63` vào 64 lát cắt vì Attention mặc định không biết thứ tự không gian (không có bước này, hai chữ số giống/lặp dễ bị đổi chỗ như `7156` $\rightarrow$ `7515`).
   - **Khởi tạo `init_h, init_c` bằng `MaxPooling1D(8)` $\rightarrow$ `Flatten()`:**
     - *Tại sao không `Flatten()` trực tiếp?* Vì `Flatten()` trên `(64, 128)` tạo ra `8,192` chiều $\rightarrow$ hơn 1 triệu tham số gây overfit nặng.
     - *Tại sao không dùng `GlobalAveragePooling1D()`?* Vì lấy trung bình toàn ảnh làm mất thứ tự trái $\rightarrow$ phải và làm nhòe dấu chấm `.` nhỏ.
     - `MaxPooling1D(8)` gom 64 lát cắt thành **8 vùng theo đúng thứ tự trái $\rightarrow$ phải** và giữ lại giá trị cực đại (nét sáng nhất như dấu `.` và `-`), vừa giữ thứ tự vừa giảm mạnh tham số.
   - **LSTM Decoder + `MultiHeadAttention(num_heads=4, key_dim=32)`:**
     - Nhúng `dec_in` (64 chiều) cộng với mã vị trí bước giải mã `0..6`, qua `LSTM(128, return_sequences=True)`.
     - `MultiHeadAttention` dùng `4` đầu chú ý song song (mỗi đầu `32` chiều): các đầu có thể chuyên trách theo dõi vị trí trái $\rightarrow$ phải, nét số, dấu chấm và dấu trừ.
     - `TimeDistributed(Dense(64, relu))` $\rightarrow$ `TimeDistributed(Dense(15, softmax))` xuất xác suất cho 15 token tại mỗi bước.
5. **Hàm `greedy_decode(...)`:**
   - Khi kiểm thử trên tập Test, mô hình **không được biết nhãn thật**. Do đó chỉ gán sẵn `dec_in[:, 0] = <SOS>`, rồi chạy vòng `for t in range(max_dec_len - 1)`: ở mỗi bước `t`, dùng `np.argmax(probs[:, t, :])` lấy token xác suất cao nhất điền vào cột `t + 1` để tiếp tục dự đoán bước kế tiếp (tự hồi quy). Cuối cùng lọc bỏ `<PAD>`, `<SOS>` và `.split('<EOS>')[0]` để cắt chuỗi ngay khi gặp `<EOS>`.
6. **Hàm `main()`:**
   - `pd.read_csv(LABELS_CSV, dtype={'label': str})`: bắt buộc ép `dtype=str` để chuỗi số như `'0123'` hay `'.4180'` không bị Pandas tự động ép thành số thực/số nguyên làm mất số `0` đầu.
   - Huấn luyện tối đa `45` epochs (`batch_size=32`, `EarlyStopping(patience=10, restore_best_weights=True)`), đạt điểm tối ưu ở Epoch `21` (`val_loss = 0.0546`, `val_accuracy = 98.01%`), cho kết quả Test **91.32% Exact-Match (`305/334`)**, **1.89% CER**, **98.11% Character Accuracy**, và **0/334 lỗi chữ số hay dấu âm**.

#### B. Câu hỏi "hỏi vặn" thường gặp cho `task3_cnn_lstm_attention.py`
- **Hỏi:** *Tại sao `val_accuracy` lúc train đạt tới `98.01%` nhưng `Exact-Match` trên tập Test lại là `91.32%`?*
  - **Trả lời:** Có 2 lý do: (1) `val_accuracy` đo trên **từng ký tự riêng lẻ** và dùng **Teacher Forcing** (được mớm ký tự đúng ở bước trước), còn `Exact-Match` yêu cầu **đúng tuyệt đối 100% cả 5–6 ký tự** trong chuỗi bằng **Greedy Decoding** tự dự đoán (sai 1 dấu chấm `.` là cả chuỗi bị tính 0 điểm). Thực tế xác suất đúng từng ký tự của mô hình trên Test là `98.11%` ($1 - \text{CER}$), và $0.9811^5 \approx 90.9\%$ — khớp hoàn toàn với `91.32%`!
- **Hỏi:** *29 lỗi còn lại trên 334 ảnh Test là lỗi gì? Tại sao lại sai?*
  - **Trả lời:** Nhóm đã kiểm tra toàn bộ 334 ảnh Test: **0/334 ảnh bị sai chữ số (0–9) hay dấu trừ (`-`)**, tức độ chính xác chữ số đạt **100%**. Toàn bộ 29 lỗi đều là **lỗi dấu chấm thập phân (`.`)** cực mờ trên nền ô LED chưa bật sáng (hoặc dấu chấm nằm trước dấu trừ như `.-9586` trên ảnh synthetic).
- **Hỏi:** *Nếu bỏ `sample_weight` ở `model.fit()` thì điều gì xảy ra?*
  - **Trả lời:** Các chuỗi dài 3–4 ký tự có nhiều ô `<PAD>` ở cuối. Không có `sample_weight` che `<PAD>`, mô hình sẽ được tính điểm oan khi đoán đúng `<PAD>`, làm `loss` giảm giả tạo và mô hình lười học các ký tự thật ở đầu chuỗi.
- **Hỏi:** *Tại sao phải dùng `Permute((2, 1, 3))` trước `Reshape`? Không gọi `Permute` mà `Reshape` thẳng có được không?*
  - **Trả lời:** Không được! Đầu ra CNN có dạng `(batch, H=8, W=64, C=128)`. Nếu `Reshape` thẳng không `Permute`, bộ nhớ sẽ gộp theo từng hàng ngang (`H` đứng trước `W`), làm trộn lẫn các chữ số bên trái với các chữ số bên phải. `Permute((2, 1, 3))` đưa trục chiều rộng `W=64` lên làm trục thứ tự đầu tiên, đảm bảo chuỗi vector đi đúng từ **trái sang phải** khớp với thứ tự đọc chữ số.
