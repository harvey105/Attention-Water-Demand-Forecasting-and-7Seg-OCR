"""Task 3 - Bước 1: Tải bộ ảnh 7SEG_OCR, trích xuất ảnh PNG và chia tập Train/Val/Test."""
import io  # Chuyển chuỗi byte nhị phân trong file Parquet thành luồng dữ liệu cho PIL đọc ảnh
import os  # Quản lý thư mục và kiểm tra sự tồn tại của file
import requests  # Tải file Parquet từ Hugging Face Hub theo từng khối (stream)
import pandas as pd  # Đọc dữ liệu Parquet và lưu bảng nhãn CSV
from PIL import Image  # Giải mã byte thành ảnh RGB và xuất ra định dạng PNG
from sklearn.model_selection import train_test_split  # Chia dữ liệu phân tầng theo độ dài chuỗi nhãn

RAW_DIR = './data/raw'
PROC_DIR = './data/processed'
PARQUET_PATH = os.path.join(RAW_DIR, '7seg_train.parquet')
IMG_DIR = os.path.join(RAW_DIR, '7seg_images')
LABELS_CSV = os.path.join(RAW_DIR, '7seg_labels.csv')
HF_PARQUET_URL = (
    'https://huggingface.co/datasets/MiXaiLL76/7SEG_OCR/'
    'resolve/main/data/train-00000-of-00001.parquet'
)


def download_parquet():
    """Tải file Parquet gốc (~192 MB) từ Hugging Face về data/raw nếu chưa có sẵn."""
    os.makedirs(RAW_DIR, exist_ok=True)  # exist_ok=True tránh lỗi khi thư mục đã được tạo
    os.makedirs(PROC_DIR, exist_ok=True)
    if not os.path.exists(PARQUET_PATH):  # Bỏ qua bước tải nếu file đã lưu cache cục bộ
        # stream=True cho phép tải dần theo từng mảnh thay vì nạp hết 192 MB vào RAM một lúc
        with requests.get(HF_PARQUET_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(PARQUET_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # Ghi từng khối 1 MB (1024*1024 bytes)
                    if chunk:  # Bỏ qua các chunk rỗng (keep-alive)
                        f.write(chunk)


def main():
    """Giải nén 3,333 ảnh PNG, trích xuất nhãn và chia tập Train (80%), Val (10%), Test (10%)."""
    download_parquet()
    os.makedirs(IMG_DIR, exist_ok=True)
    df = pd.read_parquet(PARQUET_PATH)  # Đọc bảng gồm 2 cột: 'image' (bytes) và 'text' (chuỗi nhãn)

    records = []
    # Duyệt từng dòng trong bảng Parquet để lưu thành ảnh PNG rời rạc phục vụ huấn luyện
    for idx, row in df.iterrows():
        img_obj = row['image']
        # Trong Hugging Face Parquet, ảnh thường lưu dạng dict {'bytes': ...}; xử lý cả 2 trường hợp cho an toàn
        img_bytes = img_obj['bytes'] if isinstance(img_obj, dict) else img_obj
        filename = f'img_{idx:05d}.png'  # Đặt tên chuẩn 5 chữ số (vd img_00000.png) dễ sắp xếp
        img_path = os.path.join(IMG_DIR, filename)
        if not os.path.exists(img_path):  # Chỉ giải nén nếu file ảnh chưa có, tiết kiệm I/O khi chạy lại
            Image.open(io.BytesIO(img_bytes)).convert('RGB').save(img_path)  # Ép chuẩn 3 kênh màu RGB
        records.append({'filename': filename, 'label': str(row['text']).strip()})

    meta_df = pd.DataFrame(records)
    meta_df['seq_len'] = meta_df['label'].str.len()  # Tính độ dài chuỗi (2..6 ký tự) để chia phân tầng

    # Lần 1: tách 80% Train và 20% tạm (temp_df); stratify giúp tỷ lệ chuỗi 2-6 ký tự đồng đều ở các tập
    train_df, temp_df = train_test_split(
        meta_df, test_size=0.20, random_state=42, stratify=meta_df['seq_len']
    )
    # Lần 2: chia đôi 20% tạm (test_size=0.50) thành 10% Validation và 10% Test với cùng random_state=42
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['seq_len']
    )

    # Gộp chung vào 1 file CSV kèm cột 'split' và xóa cột phụ 'seq_len'
    full_df = pd.concat([
        train_df.assign(split='train'),
        val_df.assign(split='val'),
        test_df.assign(split='test'),
    ], ignore_index=True).drop(columns=['seq_len'])
    full_df.to_csv(LABELS_CSV, index=False)  # index=False không ghi cột số thứ tự dư thừa ra file CSV

    print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')
    print('Character set:', ''.join(sorted(set(''.join(full_df['label'])))))
    print('Saved labels to:', LABELS_CSV)


if __name__ == '__main__':
    main()
