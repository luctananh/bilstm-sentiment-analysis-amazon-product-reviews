# bilstm-sentiment-analysis-amazon-product-reviews

## Giới thiệu

Dự án này xây dựng một hệ thống phân tích cảm xúc (sentiment analysis) cho các đánh giá sản phẩm trên Amazon. Hệ thống sử dụng mô hình mạng nơ-ron tái phát hai chiều với cơ chế bộ nhớ dài-ngắn hạn (BiLSTM) và biểu diễn từ Word2Vec để phân loại cảm xúc của văn bản thành "Tích cực" hoặc "Tiêu cực". Ngoài ra, dự án còn tích hợp một công cụ cào dữ liệu (scraper) từ Amazon để thu thập các đánh giá sản phẩm dựa trên URL. Cuối cùng, một giao diện người dùng đồ họa (GUI) được xây dựng bằng Streamlit cho phép người dùng tương tác với mô hình và công cụ cào dữ liệu một cách dễ dàng.

## Cài đặt chương trình

Để chạy chương trình, bạn cần thực hiện các bước sau:

1.  **Tải và cài đặt ChromeDriver:**
    * Tải ChromeDriver phù hợp với phiên bản Chrome bạn đang sử dụng từ [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads).
    * Giải nén và đặt thư mục `chromedriver-win64` vào một vị trí thích hợp (ví dụ: `D:\\code\\chuyende2\\`). **Lưu ý:** Thay đổi đường dẫn `CHROMEDRIVER_PATH` trong `app.py` nếu bạn đặt ChromeDriver ở vị trí khác.

2.  **Tạo và kích hoạt môi trường ảo (khuyến nghị):**

    ```bash
    python -m venv env
    env\Scripts\activate  # Trên Windows
    source env/bin/activate # Trên macOS và Linux
    ```

3.  **Cài đặt các thư viện cần thiết:**

    ```bash
    pip install pandas numpy torch streamlit tensorflow joblib nltk scikit-learn selenium beautifulsoup4
    ```

    * **Lưu ý:** Các thư viện cho file `main.py` đã được liệt kê trong hướng dẫn chạy file này.

## Chạy chương trình

Dự án này bao gồm hai phần chính: huấn luyện mô hình và giao diện người dùng.

### 1. Huấn luyện mô hình (`main.py`)

File `main.py` chứa code để huấn luyện mô hình BiLSTM phân tích cảm xúc.

* **Các thư viện cần thiết:**

    ```bash
    pip install pandas numpy nltk python-dotenv kagglehub tqdm joblib imbalanced-learn scikit-learn gensim torch tensorflow
    ```

* **Tải dữ liệu:** Script sử dụng dữ liệu từ Kaggle. Đảm bảo bạn đã cài đặt `kagglehub` và xác thực với tài khoản Kaggle của mình.

* **Chạy huấn luyện:**

    ```bash
    python main.py
    ```

    Script sẽ:

    * Kiểm tra GPU khả dụng.
    * Tải dữ liệu từ Kaggle.
    * Tiền xử lý văn bản (chuyển về chữ thường, loại bỏ stop words, lemmatization).
    * Huấn luyện mô hình Word2Vec để tạo embedding cho từ.
    * Xây dựng và huấn luyện mô hình BiLSTM.
    * Đánh giá mô hình trên tập kiểm tra.
    * Lưu mô hình đã huấn luyện và tokenizer vào thư mục `models`.

### 2. Giao diện người dùng (`app.py`)

File `app.py` cung cấp giao diện người dùng đồ họa (GUI) bằng Streamlit để tương tác với mô hình và công cụ cào dữ liệu.

* **Vào môi trường ảo (nếu bạn đã tạo):**

    ```bash
     env\Scripts\activate  # Trên Windows
    source env/bin/activate # Trên macOS và Linux
    ```

* **Chạy ứng dụng Streamlit:**

    ```bash
    streamlit run app.py
    ```

    Ứng dụng sẽ mở trong trình duyệt web của bạn.

#### Chức năng

Ứng dụng Streamlit cung cấp các chức năng sau:

* **Sentiment Analysis:** Nhập văn bản và dự đoán cảm xúc (tích cực hoặc tiêu cực).
* **Amazon Review Scraper:**
    * Nhập thông tin tài khoản Amazon (email, password) và URL của trang đánh giá sản phẩm.
    * Ứng dụng sẽ cào các đánh giá và phân loại cảm xúc của từng đánh giá.
* **Model Evaluation:**
    * Tải file CSV chứa dữ liệu đánh giá (với các cột yêu cầu: sentiment, id, date, query, user, text).
    * Ứng dụng sẽ đánh giá hiệu suất của mô hình trên dữ liệu đã tải.

## Cấu trúc code

* `main.py`: Chứa code huấn luyện mô hình BiLSTM.
* `app.py`: Chứa code cho ứng dụng Streamlit, bao gồm cả chức năng cào dữ liệu và phân tích cảm xúc.
* `models/`: Thư mục này chứa các file đã lưu của mô hình đã huấn luyện (`BiLSTM_Word2Vec_model.pt`) và tokenizer (`tokenizer_and_encoder.joblib`).
* `chromedriver-win64/`: Thư mục chứa ChromeDriver (nếu bạn chọn đặt nó ở đây).

## Lưu ý quan trọng

* **Bảo mật:** Khi sử dụng chức năng cào dữ liệu, hãy cẩn thận với thông tin tài khoản Amazon của bạn.
* **ChromeDriver:** Đảm bảo phiên bản ChromeDriver tương thích với phiên bản Chrome bạn đang sử dụng.
* **Hiệu suất:** Quá trình huấn luyện mô hình có thể tốn thời gian, đặc biệt nếu bạn không có GPU.
* **Yêu cầu về dữ liệu đánh giá mô hình:** File CSV dùng để đánh giá mô hình phải có các cột sau: `sentiment`, `id`, `date`, `query`, `user`, `text`. Cột `text` chứa nội dung đánh giá, cột `sentiment` chứa nhãn cảm xúc (0 cho tiêu cực, 4 cho tích cực).
