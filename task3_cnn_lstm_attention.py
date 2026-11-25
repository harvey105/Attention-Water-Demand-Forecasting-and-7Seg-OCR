"""Task 3 - Bước 2: Mô hình CNN Encoder + LSTM Decoder với Multi-Head Attention cho 7SEG_OCR."""
import os  # Thiết lập PYTHONHASHSEED và ghép đường dẫn ảnh
import random  # Cố định seed ngẫu nhiên mức Python
import numpy as np  # Tạo mảng ảnh, chuỗi vị trí và tính toán ma trận đánh giá
import pandas as pd  # Đọc bảng nhãn và thông tin phân tập train/val/test từ CSV
import tensorflow as tf  # Cố định seed TensorFlow để tái lập kết quả huấn luyện
from PIL import Image  # Mở file PNG, chuyển hệ màu RGB và resize về kích thước chuẩn
from tensorflow.keras import Model  # Functional API xây dựng kiến trúc nhiều đầu vào (ảnh + vị trí + decoder)
from tensorflow.keras.layers import (
    Input, Conv2D, Conv1D, BatchNormalization, MaxPooling2D,
    Permute, Reshape, MaxPooling1D, Flatten,
    Dense, Embedding, LSTM, MultiHeadAttention, Add, Concatenate,
    TimeDistributed, Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Dừng sớm và giảm LR tự động
from tensorflow.keras.optimizers import Adam  # Bộ tối ưu Adam cho tốc độ hội tụ ổn định

SEED = 42  # Cố định seed=42 đảm bảo tính tái lập giữa các lần chạy
IMG_DIR = './data/raw/7seg_images'
LABELS_CSV = './data/raw/7seg_labels.csv'
MODEL_PATH = './data/processed/task3_model.keras'
EVAL_PATH = './data/processed/task3_eval.npz'
IMG_H, IMG_W, ENC_STEPS = 64, 256, 64  # Giữ 64 bước ngang (256/4) để dấu '.' và '-' không bị mất khi pooling
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = '<PAD>', '<SOS>', '<EOS>'  # Các token điều khiển độ dài chuỗi biến thiên


def set_seed(seed=SEED):
    """Cố định đồng bộ random seed cho Python, NumPy và TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def levenshtein_distance(s1, s2):
    """Tính khoảng cách sửa đổi Levenshtein (thêm, xóa, thay thế) giữa chuỗi dự đoán và chuỗi thật."""
    if len(s1) < len(s2):  # Đảm bảo s1 luôn là chuỗi dài hơn để tiết kiệm bộ nhớ bảng quy hoạch động
        return levenshtein_distance(s2, s1)
    prev = list(range(len(s2) + 1))  # Hàng cơ sở: chi phí biến chuỗi rỗng thành tiền tố s2
    for i, c1 in enumerate(s1):
        curr = [i + 1]  # Chi phí xóa i+1 ký tự đầu của s1
        for j, c2 in enumerate(s2):
            # Chọn giá trị nhỏ nhất trong 3 phép: chèn (+1), xóa (+1), hoặc giữ/thay thế (c1 != c2)
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]  # Ô cuối cùng là tổng số phép sửa tối thiểu giữa hai chuỗi


def load_split_data(df, char_to_id, max_dec_len):
    """Đọc ảnh RGB chuẩn hóa [0,1] và tạo cặp chuỗi Teacher Forcing + mặt nạ bỏ qua <PAD>."""
    images, labels = [], df['label'].tolist()
    # Khởi tạo toàn bộ ma trận đích bằng ID của <PAD> để các chuỗi ngắn tự động được đệm cuối
    dec_in = np.full((len(df), max_dec_len), char_to_id[PAD_TOKEN], dtype='int32')
    dec_out = np.full((len(df), max_dec_len), char_to_id[PAD_TOKEN], dtype='int32')
    for i, row in enumerate(df.itertuples()):
        img = Image.open(os.path.join(IMG_DIR, row.filename)).convert('RGB').resize((IMG_W, IMG_H))
        images.append(np.asarray(img, dtype='float32') / 255.0)  # Chia 255.0 đưa giá trị pixel từ [0,255] về [0,1]
        seq = [char_to_id[SOS_TOKEN]] + [char_to_id[c] for c in row.label] + [char_to_id[EOS_TOKEN]]
        dec_in[i, : len(seq) - 1] = seq[:-1]   # Đầu vào Decoder dịch phải: [<SOS>, c1, ..., cn]
        dec_out[i, : len(seq) - 1] = seq[1:]   # Nhãn mục tiêu cần dự đoán: [c1, ..., cn, <EOS>]
    sample_weights = (dec_out != char_to_id[PAD_TOKEN]).astype('float32')  # Đặt trọng số=0 ở vị trí <PAD> khi tính loss
    return np.asarray(images, dtype='float32'), dec_in, dec_out, sample_weights, labels


def build_cnn_lstm_attention(vocab_size, max_dec_len):
    """Xây dựng kiến trúc CNN Encoder (8x64) + Conv1D + LSTM Decoder + Multi-Head Attention."""
    img_in = Input(shape=(IMG_H, IMG_W, 3), name='image_input')
    dec_in = Input(shape=(max_dec_len,), name='decoder_input')
    enc_pos_in = Input(shape=(ENC_STEPS,), dtype='int32', name='enc_pos_input')
    dec_pos_in = Input(shape=(max_dec_len,), dtype='int32', name='dec_pos_input')

    # 4 khối CNN trích xuất đặc trưng ảnh: giữ chiều cao H=8 và rộng W=64 (chỉ pool ngang 2 lần -> 4x)
    x = img_in
    for filters, pool in [(48, (2, 2)), (64, (2, 2)), (128, (2, 1)), (128, None)]:
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)  # padding='same' giữ nguyên biên ảnh
        x = BatchNormalization()(x)  # Chuẩn hóa batch giúp hội tụ nhanh và ổn định gradient
        if pool is not None:  # Khối cuối không pool để bảo toàn độ nét của dấu chấm thập phân '.'
            x = MaxPooling2D(pool)(x)

    # Đảo trục (H=8, W=64, C=128) -> (W=64, H=8, C=128) rồi gộp H*C thành chuỗi 64 lát cắt trái->phải
    x = Reshape((ENC_STEPS, 8 * 128))(Permute((2, 1, 3))(x))
    enc_seq = Dense(128, activation='relu')(x)  # Giảm chiều mỗi lát cắt từ 1024 xuống 128
    enc_seq = Conv1D(128, kernel_size=3, padding='same', activation='relu')(enc_seq)  # Học ngữ cảnh 3 lát cắt kề nhau
    enc_seq_pos = Add()([enc_seq, Embedding(ENC_STEPS, 128)(enc_pos_in)])  # Cộng thêm mã vị trí 0..63 cho Attention

    # MaxPooling1D(8) giữ lại 8 vùng theo thứ tự trái->phải và bảo toàn tín hiệu mạnh ('.', '-') tránh overfit
    enc_summary = Dropout(0.2)(Flatten()(MaxPooling1D(pool_size=8)(enc_seq_pos)))
    init_h, init_c = Dense(128, activation='tanh')(enc_summary), Dense(128, activation='tanh')(enc_summary)

    # Nhúng ký tự kết hợp mã vị trí bước giải mã 0..max_dec_len-1 trước khi đưa vào LSTM Decoder
    dec_in_seq = Add()([Embedding(vocab_size, 64)(dec_in), Embedding(max_dec_len, 64)(dec_pos_in)])
    dec_out = LSTM(128, return_sequences=True)(dec_in_seq, initial_state=[init_h, init_c])

    # Multi-Head Attention (4 đầu x 32 chiều) giúp Decoder vừa đọc ký tự vừa bám sát vị trí trên ảnh
    context = MultiHeadAttention(num_heads=4, key_dim=32)(query=dec_out, key=enc_seq_pos, value=enc_seq_pos)
    combined = Dropout(0.2)(Concatenate(axis=-1)([dec_out, context]))  # Ghép đầu ra LSTM và ngữ cảnh Attention
    outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(TimeDistributed(Dense(64, activation='relu'))(combined))

    model = Model(inputs=[img_in, dec_in, enc_pos_in, dec_pos_in], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', weighted_metrics=['accuracy'])
    return model


def make_pos(n_samples, max_dec_len):
    """Tạo mảng chỉ số vị trí [0..ENC_STEPS-1] và [0..max_dec_len-1] lặp cho n_samples mẫu."""
    return np.tile(np.arange(ENC_STEPS, dtype='int32'), (n_samples, 1)), np.tile(np.arange(max_dec_len, dtype='int32'), (n_samples, 1))


def greedy_decode(model, X_img, char_to_id, id_to_char, max_dec_len):
    """Giải mã tự hồi quy (từng bước) trên tập Test mà không dùng nhãn thật."""
    n = len(X_img)
    dec_in = np.full((n, max_dec_len), char_to_id[PAD_TOKEN], dtype='int32')
    dec_in[:, 0] = char_to_id[SOS_TOKEN]  # Bước t=0 luôn mồi bằng token bắt đầu <SOS>
    enc_pos, dec_pos = make_pos(n, max_dec_len)
    for t in range(max_dec_len - 1):  # Lần lượt dự đoán ký tự tại bước t để đưa làm đầu vào cho bước t+1
        probs = model.predict([X_img, dec_in, enc_pos, dec_pos], batch_size=128, verbose=0)
        dec_in[:, t + 1] = np.argmax(probs[:, t, :], axis=-1)  # Chọn ID có xác suất cao nhất (greedy)
    pred_ids = np.argmax(model.predict([X_img, dec_in, enc_pos, dec_pos], batch_size=128, verbose=0), axis=-1)
    preds = []
    for row in pred_ids:  # Ghép ID thành chuỗi và cắt ngay tại ký tự kết thúc <EOS>
        chars = [id_to_char[int(i)] for i in row if i not in (char_to_id[PAD_TOKEN], char_to_id[SOS_TOKEN])]
        preds.append(''.join(chars).split(EOS_TOKEN)[0])  # Bỏ phần dư phía sau <EOS>
    return preds


def main():
    """Huấn luyện CNN-LSTM-Attention, đánh giá Exact-Match/CER trên Test và lưu kết quả."""
    set_seed()
    df = pd.read_csv(LABELS_CSV, dtype={'label': str}).fillna('')  # dtype=str giữ nguyên chuỗi như '007'
    chars = sorted(set(''.join(df['label'])))  # Bộ ký tự xuất hiện trong dữ liệu: '-.0123456789'
    vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + chars  # Tổng 15 token (3 token điều khiển + 12 ký tự)
    char_to_id, id_to_char = {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}
    max_dec_len = int(df['label'].str.len().max()) + 1  # Độ dài tối đa (6) + 1 cho <SOS>/<EOS> = 7 bước

    X_tr, di_tr, do_tr, sw_tr, _ = load_split_data(df[df['split'] == 'train'], char_to_id, max_dec_len)
    X_val, di_val, do_val, sw_val, _ = load_split_data(df[df['split'] == 'val'], char_to_id, max_dec_len)
    test_df = df[df['split'] == 'test']
    X_te, _, _, _, y_te_str = load_split_data(test_df, char_to_id, max_dec_len)
    ep_tr, dp_tr = make_pos(len(X_tr), max_dec_len)
    ep_val, dp_val = make_pos(len(X_val), max_dec_len)

    model = build_cnn_lstm_attention(len(vocab), max_dec_len)
    model.summary()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),  # Khôi phục trọng số tốt nhất
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5),  # Giảm một nửa LR khi chững
    ]
    history = model.fit(
        [X_tr, di_tr, ep_tr, dp_tr], do_tr, sample_weight=sw_tr,  # sample_weight bỏ qua vị trí <PAD>
        validation_data=([X_val, di_val, ep_val, dp_val], do_val, sw_val),
        epochs=45, batch_size=32, callbacks=callbacks, verbose=1,
    )
    model.save(MODEL_PATH)

    y_pred_str = greedy_decode(model, X_te, char_to_id, id_to_char, max_dec_len)
    exact_matches = sum(p == t for p, t in zip(y_pred_str, y_te_str))  # Đếm số ảnh khớp hoàn toàn cả chuỗi
    cer = sum(levenshtein_distance(p, t) for p, t in zip(y_pred_str, y_te_str)) / sum(len(t) for t in y_te_str)
    print(f'\n=== Task 3 Evaluation on Test Set ({len(y_te_str)} samples) ===')
    print(f'Exact-Match Seq Accuracy: {exact_matches / len(y_te_str) * 100:.2f}% ({exact_matches}/{len(y_te_str)})')
    print(f'Character Error Rate(CER): {cer * 100:.2f}% | Char Accuracy: {(1.0 - cer) * 100:.2f}%')

    np.savez(
        EVAL_PATH, y_true=np.array(y_te_str), y_pred=np.array(y_pred_str),
        filenames=test_df['filename'].values,
        loss=np.array(history.history['loss']), val_loss=np.array(history.history['val_loss']),
        accuracy=np.array(history.history['accuracy']), val_accuracy=np.array(history.history['val_accuracy']),
    )
    print('Saved evaluation outputs to:', EVAL_PATH)


if __name__ == '__main__':
    main()
