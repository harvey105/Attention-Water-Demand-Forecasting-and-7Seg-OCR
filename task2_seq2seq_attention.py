"""Task 2 - Bước 2: Huấn luyện mô hình Seq2Seq LSTM Encoder-Decoder + Attention dự báo 5 giờ tới."""
import os  # Thiết lập biến môi trường PYTHONHASHSEED cho tính tái lập
import random  # Cố định bộ sinh số ngẫu nhiên chuẩn của Python
import numpy as np  # Đọc/lưu file .npz và tính toán sai số đánh giá
import tensorflow as tf  # Thư viện Deep Learning chính, cố định seed TensorFlow
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error  # Các metric chuẩn của Task 2
from tensorflow.keras import Model  # Keras Functional API cho phép nối nhánh Attention linh hoạt
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, Attention, Concatenate,
    TimeDistributed, Dense, Dropout, Reshape,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Dừng sớm chống overfit & giảm LR khi bão hòa
from tensorflow.keras.optimizers import Adam  # Bộ tối ưu Adaptive Moment phù hợp dữ liệu chuỗi thời gian

SEED = 42  # Hằng số seed=42 đảm bảo mỗi lần chạy lại cho kết quả ổn định, đồng nhất
DATA_PATH = './data/processed/task2_data.npz'
MODEL_PATH = './data/processed/task2_model.keras'
EVAL_PATH = './data/processed/task2_eval.npz'


def set_seed(seed=SEED):
    """Cố định đồng bộ trạng thái ngẫu nhiên cho Python, NumPy và TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)  # Cố định hàm băm (hash) nội bộ của Python
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def nse_score(y_true, y_pred):
    """Tính chỉ số Nash-Sutcliffe Efficiency: NSE = 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)."""
    num = np.sum((y_true - y_pred) ** 2)  # Tổng bình phương sai số dự báo (SS_res)
    den = np.sum((y_true - np.mean(y_true)) ** 2)  # Tổng bình phương phương sai dữ liệu thực (SS_tot)
    return 1.0 - (num / den)  # NSE = 1 là hoàn hảo; > 0.90 là rất xuất sắc trong thủy văn/cấp nước


def evaluate_forecasts(y_true, y_pred):
    """Tính và in bảng 4 chỉ số R2, NSE, MAE, RMSE tổng thể và riêng từng bước t+1..t+5."""
    print('\n=== Task 2 Evaluation on Test Set (m3/h) ===')
    yt_flat, yp_flat = y_true.reshape(-1), y_pred.reshape(-1)  # Duỗi phẳng (N, 5) -> (N*5,) để tính tổng thể
    print(
        f'Overall (t+1..t+5) | R2: {r2_score(yt_flat, yp_flat):.4f} | '
        f'NSE: {nse_score(yt_flat, yp_flat):.4f} | '
        f'MAE: {mean_absolute_error(yt_flat, yp_flat):.2f} | '
        f'RMSE: {root_mean_squared_error(yt_flat, yp_flat):.2f}'
    )
    # Lặp qua từng bước tương lai (0->t+1 .. 4->t+5) để kiểm tra độ suy giảm sai số theo thời gian
    for step in range(y_true.shape[1]):
        yt_s, yp_s = y_true[:, step], y_pred[:, step]
        print(
            f'Horizon t+{step + 1}        | R2: {r2_score(yt_s, yp_s):.4f} | '
            f'NSE: {nse_score(yt_s, yp_s):.4f} | '
            f'MAE: {mean_absolute_error(yt_s, yp_s):.2f} | '
            f'RMSE: {root_mean_squared_error(yt_s, yp_s):.2f}'
        )


def build_seq2seq_attention(n_steps, n_features, n_pred):
    """Xây dựng kiến trúc LSTM Encoder-Decoder kết hợp Attention (Luong-style dot-product)."""
    inputs = Input(shape=(n_steps, n_features))  # Đầu vào dạng (batch, 48, 9)

    # return_sequences=True: xuất cả 48 hidden states làm Key/Value cho Attention
    # return_state=True: xuất thêm trạng thái cuối (state_h, state_c) để khởi tạo Decoder
    enc_out, state_h, state_c = LSTM(96, return_sequences=True, return_state=True)(inputs)

    # Nhân bản vector tóm tắt state_h thành 5 bước (batch, 5, 96) làm chuỗi đầu vào cho Decoder
    dec_in = RepeatVector(n_pred)(state_h)
    # return_sequences=True: xuất đủ 5 trạng thái ẩn đầu ra ứng với t+1..t+5; initial_state nhận ngữ cảnh Encoder
    dec_out = LSTM(96, return_sequences=True)(dec_in, initial_state=[state_h, state_c])

    # Attention([query, value]): mỗi bước Decoder tự tìm giờ quá khứ liên quan nhất trong 48h Encoder
    context = Attention()([dec_out, enc_out])
    combined = Concatenate(axis=-1)([dec_out, context])  # Ghép trên chiều đặc trưng -> (batch, 5, 192)

    # TimeDistributed áp dụng cùng một lớp Dense cho từng bước trong 5 bước tương lai (chia sẻ trọng số)
    x = TimeDistributed(Dense(32, activation='relu'))(combined)
    x = Dropout(0.1)(x)  # Tắt ngẫu nhiên 10% neuron khi train để giảm overfitting
    outputs = Reshape((n_pred,))(TimeDistributed(Dense(1))(x))  # Đưa từ (batch, 5, 1) về (batch, 5)

    model = Model(inputs=inputs, outputs=outputs)
    # Adam lr=1e-3 là tốc độ học chuẩn; loss='mse' tối ưu trực tiếp cho RMSE/R2/NSE
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    return model


def main():
    """Nạp dữ liệu chuẩn hóa, huấn luyện mô hình, quy đổi đơn vị thực và lưu kết quả."""
    set_seed()
    data = np.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    target_mean, target_std = float(data['target_mean'][0]), float(data['target_std'][0])

    model = build_seq2seq_attention(X_train.shape[1], X_train.shape[2], y_train.shape[1])
    model.summary()

    callbacks = [
        # patience=7: dừng nếu 7 epoch liên tiếp val_loss không giảm; restore_best_weights=True khôi phục bản tốt nhất
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        # factor=0.5: giảm một nửa learning rate nếu val_loss chững lại 3 epoch, xuống tối thiểu 1e-5
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=40, batch_size=64, callbacks=callbacks, verbose=1,  # batch_size=64 cân bằng tốc độ & độ mượt gradient
    )
    model.save(MODEL_PATH)  # Lưu mô hình tối ưu nhất ra file .keras

    # Nghịch đảo z-score: x_thực = x_chuẩn_hóa * std + mean để đánh giá trên đơn vị m3/h
    y_pred = model.predict(X_test, batch_size=256) * target_std + target_mean
    y_true = y_test * target_std + target_mean
    evaluate_forecasts(y_true, y_pred)

    # Xuất lịch sử học và kết quả dự báo ra file .npz để generate_figures.py vẽ biểu đồ độc lập
    np.savez(
        EVAL_PATH, y_true=y_true, y_pred=y_pred,
        loss=np.array(history.history['loss']),
        val_loss=np.array(history.history['val_loss']),
        mae=np.array(history.history['mae']),
        val_mae=np.array(history.history['val_mae']),
    )
    print('Saved evaluation outputs to:', EVAL_PATH)


if __name__ == '__main__':
    main()
