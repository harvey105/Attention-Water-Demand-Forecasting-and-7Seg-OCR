"""Task 2 - Bước 1: Tải, tạo đặc trưng thời gian và chuẩn hóa dữ liệu nhu cầu nước theo giờ."""
import os  # Tạo thư mục và nối đường dẫn file an toàn trên mọi hệ điều hành
import requests  # Gửi HTTP GET để tải trực tiếp file CSV từ kho Zenodo
import numpy as np  # Xử lý mảng số học đa chiều và tính hàm lượng giác sin/cos
import pandas as pd  # Đọc dữ liệu bảng CSV và thao tác cột thời gian (datetime)
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa z-score (mean=0, std=1) giúp LSTM hội tụ nhanh

RAW_DIR = './data/raw'  # Thư mục lưu dữ liệu gốc chưa qua xử lý
PROC_DIR = './data/processed'  # Thư mục lưu tensor .npz đã xử lý cho mô hình
CSV_PATH = os.path.join(RAW_DIR, 'water_demand.csv')
OUT_PATH = os.path.join(PROC_DIR, 'task2_data.npz')
ZENODO_URL = (
    'https://zenodo.org/api/records/11045013/files/'
    'waterconsumption_noNaN_fixedTimechange.csv/content'
)


def download_dataset():
    """Tải file CSV gốc từ Zenodo vào data/raw nếu máy chưa có sẵn file."""
    os.makedirs(RAW_DIR, exist_ok=True)  # exist_ok=True: không báo lỗi nếu thư mục đã tồn tại
    os.makedirs(PROC_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):  # Chỉ tải khi chưa có file local để tránh tải lại tốn thời gian
        resp = requests.get(ZENODO_URL, timeout=60)  # timeout=60s: tránh treo vô hạn khi mạng chậm
        resp.raise_for_status()  # Ném lỗi ngay nếu HTTP status khác 200 (link hỏng/mất mạng)
        with open(CSV_PATH, 'wb') as f:  # Mở chế độ 'wb' (write binary) để ghi nguyên vẹn byte tải về
            f.write(resp.content)


def create_sequences(values, target_col, n_steps, n_pred):
    """Cắt chuỗi thành cửa sổ trượt: đầu vào X (n_steps=48h quá khứ) và đầu ra y (n_pred=5h tới)."""
    X, y = [], []
    # Duyệt chỉ số i sao cho cửa sổ [i : i + n_steps + n_pred] không vượt quá độ dài chuỗi
    for i in range(len(values) - (n_steps + n_pred) + 1):
        X.append(values[i : i + n_steps, :])  # Lấy tất cả đặc trưng (:) trong 48 giờ liên tiếp
        y.append(values[i + n_steps : i + n_steps + n_pred, target_col])  # Lấy 5 giá trị mục tiêu kế tiếp
    return np.asarray(X, dtype='float32'), np.asarray(y, dtype='float32')  # float32 giảm một nửa RAM so với float64


def main():
    """Tiền xử lý, chia tập Train/Val/Test theo trình tự thời gian và xuất file .npz."""
    download_dataset()
    df = pd.read_csv(CSV_PATH)  # Đọc 46,032 dòng dữ liệu nhu cầu nước theo giờ

    # Trích xuất giờ bắt đầu (vd " 00:00-01:00 " -> bỏ khoảng trắng -> lấy 2 ký tự đầu -> 0) và thứ trong tuần
    df['Date_dt'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')  # Ép đúng định dạng tháng/ngày/năm
    df['Hour'] = df['Time'].astype(str).str.strip().str.slice(0, 2).astype(int)
    df['DayOfWeek'] = df['Date_dt'].dt.day_of_week  # Thứ 2=0 .. Chủ nhật=6
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype('float32')  # 1.0 nếu Thứ 7/CN, 0.0 nếu ngày thường

    # Mã hóa chu kỳ tròn (2π * t / chu_kỳ) để giờ 23 và giờ 0 (hoặc CN và T2) liền kề nhau liên tục
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    df['dow_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['Rain'] = df['Rain'].fillna(0.0)  # Điền 25 ô Rain bị thiếu bằng 0.0 (mặc định không mưa)

    feature_cols = [
        'Consumption', 'Temperature', 'Rain', 'Sun',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'IsWeekend',
    ]
    values = df[feature_cols].values.astype('float32')

    # Chia theo đúng thứ tự thời gian (70% Train, 15% Val, 15% Test), KHÔNG shuffle để tránh rò rỉ tương lai
    n_total = len(values)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    train_raw = values[:n_train]
    val_raw = values[n_train : n_train + n_val]
    test_raw = values[n_train + n_val :]

    # Chỉ fit (tính mean, std) trên tập Train rồi mới transform sang Val/Test để chống data leakage
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    n_steps = 48  # Lấy 48 giờ quá khứ (2 chu kỳ ngày-đêm) giúp Encoder nắm bắt nhịp sinh hoạt hôm qua & hôm kia
    n_pred = 5    # Dự báo 5 giờ liên tiếp (t+1 .. t+5) theo đúng yêu cầu Task 2
    target_col = 0  # Vị trí cột mục tiêu 'Consumption' (index 0 trong feature_cols)

    X_train, y_train = create_sequences(train_scaled, target_col, n_steps, n_pred)
    X_val, y_val = create_sequences(val_scaled, target_col, n_steps, n_pred)
    X_test, y_test = create_sequences(test_scaled, target_col, n_steps, n_pred)

    print(f'Total usable rows: {n_total} | Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
    np.savez(
        OUT_PATH,
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        # Lưu lại mean và std của cột Consumption để quy đổi ngược dự báo về đơn vị thực m3/h
        target_mean=np.array([scaler.mean_[target_col]], dtype='float32'),
        target_std=np.array([scaler.scale_[target_col]], dtype='float32'),
    )
    print('Saved processed data to:', OUT_PATH)


if __name__ == '__main__':
    main()
