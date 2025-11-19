# Vietnamese News NLP Portal (PhoBERT)

Hệ thống web phân loại và tóm tắt tin tức tiếng Việt, xây dựng trên mô hình **PhoBERT** kết hợp:

- Backend **FastAPI** + PostgreSQL
- Bộ thu thập tin **multi-source crawler**
- Mô hình tóm tắt tiếng Việt (VietAI T5)
- Frontend web cho tra cứu, khám phá chủ đề và xem thống kê.

---

## 1. Cấu trúc dự án

```text
Project/
├── app/                      # Backend FastAPI
│   ├── main.py               # Điểm vào FastAPI, endpoints & scheduler
│   ├── schemas.py            # Pydantic schemas (Article, Stats, ...)
│   ├── crud.py               # Hàm truy vấn / ghi dữ liệu
│   ├── models.py             # SQLAlchemy models (Article, ArticlePrediction, ...)
│   ├── database.py           # Kết nối PostgreSQL
│   ├── config.py             # Cấu hình (DATABASE_URL, AUTO_CRAWL_*, ...)
│   └── services/
│       ├── classifier.py     # PhoBERT inference service
│       ├── summarizer.py     # Dịch vụ tóm tắt VietAI T5
│       ├── multi_source_crawler.py  # Crawl nhiều nguồn RSS/HTML
│       └── feed_sources.py   # Danh sách nguồn RSS tiếng Việt
├── frontend/
│   ├── index.html            # Giao diện web chính
│   ├── styles.css            # CSS cho layout & cards
│   └── app.js                # Logic gọi API, render UI
├── config/
│   └── topics.py             # 10 chủ đề mục tiêu và mapping từ dataset gốc
├── models/
│   └── best_model/           # Mô hình PhoBERT đã fine-tune (sau khi train)
├── Dataset/                  # Dữ liệu gốc / đã xử lý (tuỳ chỉnh)
├── reports/                  # Classification report, confusion matrix, ...
├── train_phobert.py          # Script huấn luyện PhoBERT
├── prepare_topics_dataset.py # Chuẩn hoá dataset sang 10 chủ đề
├── predict.py                # Script demo phân loại từ mô hình đã train
├── requirements.txt          # Thư viện Python
├── run_app.ps1               # Script chạy backend + mở frontend một lần
└── README.md                 # File mô tả này
```

---

## 2. Cài đặt & chạy hệ thống

### 2.1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2.2. Chuẩn bị database PostgreSQL

Tạo database, ví dụ:

- Host: `localhost`
- Port: `6699`
- DB: `news_nlp`
- User: `postgres`
- Password: `1234`

Sau đó điều chỉnh chuỗi kết nối trong **`run_app.ps1`** nếu cần:

```powershell
$env:DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:6699/news_nlp"
```

### 2.3. Chạy web app bằng một lệnh

Trên Windows, chạy PowerShell ở thư mục `Project/`:

```powershell
.\run_app.ps1
```

Script sẽ:

- Set các biến môi trường (`DATABASE_URL`, `AUTO_CRAWL_*`, ...)
- Mở một cửa sổ PowerShell mới và chạy backend:
  - `uvicorn app.main:app --reload --port 8000`
- Mở `frontend/index.html` trong trình duyệt làm giao diện chính.

Backend API có thể truy cập tại: `http://127.0.0.1:8000` (Swagger: `/docs`).

---

## 3. Chức năng chính của web app

### 3.1. Khám phá theo chủ đề

- Dropdown chọn **10 chủ đề tin tức** đã chuẩn hoá.
- Mỗi chủ đề hiển thị lưới card (4 ô/ hàng nếu đủ rộng):
  - Dải màu tiêu đề + badge chủ đề
  - Tiêu đề bài viết
  - Tóm tắt ngắn (generated bởi summarizer)
  - Link "Đọc chi tiết" mở sang trang báo gốc.

### 3.2. Phân loại nhanh

- Nhập nội dung hoặc đoạn tin bất kỳ.
- Gọi endpoint `/classify` của PhoBERT.
- Trả về:
  - Chủ đề dự đoán
  - Độ tin cậy và top-k dự đoán.

### 3.3. Danh sách tin đã lưu

- Lấy dữ liệu qua `/news` với phân trang (mặc định 10 tin / trang).
- Hỗ trợ lọc theo:
  - `topic` (chủ đề)
  - `source` (nguồn báo)
  - `query` (từ khoá trong tiêu đề/nội dung)
- Mỗi bản ghi hiển thị:
  - Tiêu đề, nguồn, link gốc
  - Chủ đề đã phân loại, trích đoạn nội dung.

### 3.4. Thêm bài viết thủ công

- Form nhập **tiêu đề / nguồn / URL / nội dung**.
- Tuỳ chọn "Tự động phân loại khi lưu".
- Gọi `POST /news` để lưu vào database + chạy PhoBERT nếu bật classify.

### 3.5. Thống kê Hot Topics

- Endpoint `/stats` trả về:
  - **topics**: chủ đề xuất hiện nhiều nhất trong khoảng `hours` gần đây.
  - **sources**: nguồn báo có nhiều bài nhất.
  - **source_topics**: với mỗi nguồn, chủ đề được đăng nhiều nhất.
- Frontend hiển thị 3 danh sách tương ứng.

### 3.6. Thống kê chủ đề theo từng ngày

- Endpoint `/stats/daily-topics`:
  - Với mỗi ngày trong `days` gần nhất (mặc định 7 ngày), trả về chủ đề có số bài cao nhất.
- Frontend hiển thị danh sách:
  - `YYYY-MM-DD: <chủ đề> (số bài)`
  - Giúp quan sát xu hướng chủ đề theo từng ngày.

---

## 4. Pipeline thu thập & xử lý tin tức

### 4.1. Multi-source crawler

- Định nghĩa nguồn RSS trong `app/services/feed_sources.py` (VnExpress, Tuổi Trẻ, Thanh Niên, ...).
- `MultiSourceCrawler` thực hiện:
  1. Đọc RSS, lấy URL bài viết mới.
  2. Crawl nội dung đầy đủ.
  3. Kiểm tra trùng bằng URL.
  4. Gửi sang summarizer & classifier (tuỳ cấu hình) và lưu vào DB.

### 4.2. Summarizer (VietAI T5)

- `app/services/summarizer.py` dùng model `VietAI/vit5-base-vietnews-summarization` (mặc định).
- Sinh tóm tắt ngắn tiếng Việt cho mỗi bài báo.

### 4.3. PhoBERT classifier

- `app/services/classifier.py` nạp mô hình từ `models/best_model/` (đã fine-tune).
- Cung cấp hàm `classify(text, top_k)` trả về:
  - Chủ đề dự đoán chính.
  - Danh sách các chủ đề cùng xác suất.

### 4.4. Auto-crawl với APScheduler

- Trong `app/main.py`, khi khởi động ứng dụng sẽ gọi `_start_auto_crawl_scheduler()`.
- Cấu hình qua biến môi trường (set trong `run_app.ps1`):

```powershell
$env:AUTO_CRAWL_ENABLED = "true"              # Bật/tắt auto-crawl
$env:AUTO_CRAWL_INTERVAL_MINUTES = "15"       # Chu kỳ crawl (phút)
$env:AUTO_CRAWL_LIMIT_PER_FEED = "3"          # Số bài mỗi nguồn / lần
```

Scheduler sẽ định kỳ gọi crawler, tóm tắt, phân loại và lưu bài mới.

---

## 5. Huấn luyện lại PhoBERT

### 5.1. Chuẩn hoá dữ liệu theo 10 chủ đề

- `config/topics.py`: định nghĩa 10 chủ đề mục tiêu và mapping từ label gốc.
- `prepare_topics_dataset.py`: đọc dataset ban đầu, ánh xạ sang 10 chủ đề, lưu lại file mới để train.

### 5.2. Huấn luyện

```bash
python train_phobert.py
```

Script sẽ:

- Load dữ liệu đã chuẩn hoá.
- Fine-tune PhoBERT trên tập train/val/test.
- Lưu mô hình tốt nhất vào `models/best_model/`.
- Sinh **classification report** (JSON/CSV) và **confusion matrix** (PNG) trong `reports/`.

Tuỳ chọn hyperparameter (epochs, batch size, learning rate, max_length) chỉnh trong `train_phobert.py`.

### 5.3. Chi tiết mô hình PhoBERT

- **Backbone**: `vinai/phobert-base` (12 layer Transformer, hidden size 768, pre-train trên ~20GB tiếng Việt).
- **Tokenizer**: BPE tokenizer của PhoBERT, xử lý text tiếng Việt đã được chuẩn hoá.
- **Đầu ra mô hình**:
  - Lấy vector của token `[CLS]` làm biểu diễn câu.
  - Qua một **Classification Head** (Linear + Softmax) để dự đoán xác suất cho **10 chủ đề**.
- **Nhãn (labels)**: 10 chủ đề được định nghĩa trong `config/topics.py` (Thời sự – Chính trị, Kinh tế – Tài chính, Khoa học – Công nghệ, ...).
- **Hàm mất mát**: Cross-entropy giữa phân phối dự đoán và nhãn thật.
- **Tối ưu hoá**: AdamW + scheduler (warmup, giảm learning rate theo bước).
- **Chỉ số đánh giá**:
  - Accuracy, F1 macro/micro cho toàn bộ 10 class.
  - Precision/Recall theo từng chủ đề.
  - Ma trận nhầm lẫn để xem chủ đề nào dễ bị nhầm lẫn với nhau.
- **Sử dụng trong hệ thống**:
  - Backend nạp model đã train từ `models/best_model/` thông qua `classifier.py`.
  - Endpoint `/classify` trả về:
    - `predicted_topic`: chủ đề có xác suất cao nhất.
    - `confidence`: xác suất tương ứng.
    - `top_predictions`: top-k chủ đề với xác suất.
  - Khi crawler lưu bài, PhoBERT được gọi để gắn `topic` và `confidence` vào mỗi bản ghi.

---

## 6. Một số lưu ý triển khai

- Mô hình PhoBERT và VietAI T5 khá nặng, khuyến nghị dùng GPU khi huấn luyện / tóm tắt nhiều.
- Nếu chạy trên máy yếu:
  - Giảm kích thước dataset khi huấn luyện.
  - Giảm tần suất auto-crawl hoặc giới hạn số bài mỗi lần.
- Khi deploy production, nên:
  - Đưa FastAPI + DB lên server/cloud.
  - Serve frontend từ cùng domain để tránh CORS.

---

## 7. Tham khảo

- PhoBERT: https://github.com/VinAIResearch/PhoBERT
- Transformers: https://huggingface.co/transformers/
- VietAI T5: https://huggingface.co/VietAI/vit5-base-vietnews-summarization
