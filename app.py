import sys
import os
import time
import re
import numpy as np
import pandas as pd
import torch
import streamlit as st
import tensorflow.keras as keras
from joblib import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Phục vụ cho Keras
sys.modules['keras'] = keras
sys.modules['keras.src'] = keras
sys.modules['keras.src.legacy'] = keras

# Các thư viện cho Selenium và BeautifulSoup (Amazon Scraper)
from selenium import webdriver as wd
from selenium.webdriver.chrome.service import Service  
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs

# ------------------- PHẦN SENTIMENT ANALYSIS -------------------

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text, use_lemmatization=True):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

@st.cache_resource
def load_sentiment_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer, label_encoder, word2vec_model = load("models/tokenizer_and_encoder.joblib")
    except Exception as e:
        st.error(f"Lỗi tải tokenizer và encoder: {e}")
        return None, None, None, None, None

    vocab_size = len(word2vec_model.wv.key_to_index) + 1
    embedding_dim = 100
    lstm_units = 128
    max_len = 50
    num_classes = len(label_encoder.classes_)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i, word in enumerate(word2vec_model.wv.key_to_index):
        embedding_matrix[i + 1] = word2vec_model.wv[word]

    import torch.nn as nn
    class BiLSTMModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, lstm_units, num_classes, embedding_matrix, max_len):
            super(BiLSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False
            self.lstm = nn.LSTM(embedding_dim, lstm_units, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(lstm_units * 2, num_classes)
        
        def forward(self, x):
            x = self.embedding(x)
            output, (hn, cn) = self.lstm(x)
            forward_hidden = hn[-2, :, :]
            backward_hidden = hn[-1, :, :]
            hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
            hidden = self.dropout(hidden)
            out = self.fc(hidden)
            return out

    model = BiLSTMModel(vocab_size, embedding_dim, lstm_units, num_classes, embedding_matrix, max_len)
    try:
        model.load_state_dict(torch.load("models/BiLSTM_Word2Vec_model.pt", map_location=device))
    except Exception as e:
        st.error(f"Lỗi tải model: {e}")
        return None, None, None, None, None
    model = model.to(device)
    return model, tokenizer, label_encoder, device, max_len

model, tokenizer, label_encoder, device, max_len = load_sentiment_model()
if model is None:
    st.error("Không tải được model sentiment, vui lòng kiểm tra lại các file đã lưu.")

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    input_tensor = torch.tensor(padded_sequence, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, prediction = torch.max(outputs, 1)
        prediction = prediction.cpu().numpy()[0]
    
    try:
        original_sentiment = label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
        original_sentiment = prediction

    if isinstance(original_sentiment, str):
        normalized = original_sentiment.strip().lower()
        if normalized in ["negative", "neg", "-1", "0"]:
            predicted_sentiment = "Tiêu cực"
        elif normalized in ["positive", "pos", "1", "4"]:
            predicted_sentiment = "Tích cực"
        else:
            predicted_sentiment = original_sentiment
    else:
        sentiment_map = {-1: "Tiêu cực", 0: "Tiêu cực", 1: "Tích cực", 4: "Tích cực"}
        predicted_sentiment = sentiment_map.get(original_sentiment, "Không xác định")
    return predicted_sentiment

# ------------------- PHẦN AMAZON REVIEW SCRAPER -------------------

CHROMEDRIVER_PATH = r'D:\code\chuyende2\chromedriver-win64\chromedriver.exe'

def amazon_login(driver, username, password):
    login_url = ("https://www.amazon.com/ap/signin?"
                 "openid.pape.max_auth_age=3600&"
                 "openid.return_to=https%3A%2F%2Fwww.amazon.com%2Fproduct-reviews%2FB08PZJN7BD%2Fref%3Dcm_cr_arp_d_viewopt_srt%3Fie%3DUTF8%26reviewerType%3Dall_reviews%26sortBy%3Drecent%26pageNumber%3D1&"
                 "openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&"
                 "openid.assoc_handle=usflex&"
                 "openid.mode=checkid_setup&"
                 "language=en_US&"
                 "openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&"
                 "openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0")
    driver.get(login_url)
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "ap_email")))
    email_field = driver.find_element(By.ID, "ap_email")
    email_field.clear()
    email_field.send_keys(username)
    driver.find_element(By.ID, "continue").click()
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "ap_password")))
    password_field = driver.find_element(By.ID, "ap_password")
    password_field.clear()
    password_field.send_keys(password)
    driver.find_element(By.ID, "signInSubmit").click()
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "nav-link-accountList")))
    print("Đăng nhập thành công!")

def amazon_one_review_per_rating(review_url, email, password):
    """
    Lấy 1 review của mỗi loại đánh giá (5 sao đến 1 sao) cho sản phẩm dựa vào link review.
    Trích xuất ASIN từ link sau đó xây dựng URL cho từng bộ lọc.
    """
    match = re.search(r'/product-reviews/([^/]+)', review_url)
    if not match:
        raise ValueError("Không tìm thấy ASIN trong link đã nhập!")
    asin = match.group(1)
    
    options = Options()
    # options.add_argument("--headless")
    options.add_argument("--window-size=1920x1080")
    
    user_agent = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.5735.90 Safari/537.36")
    options.add_argument(f"user-agent={user_agent}")

    service = Service(executable_path=CHROMEDRIVER_PATH)
    driver = wd.Chrome(service=service, options=options)
    
    amazon_login(driver, email, password)
    
    rating_filters = ["five_star", "four_star", "three_star", "two_star", "one_star"]
    results = []
    
    base_url = f"https://www.amazon.com/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_srt?"
    common_params = "ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"
    
    for rating_filter in rating_filters:
        url = f"{base_url}{common_params}&filterByStar={rating_filter}"
        driver.get(url)
        print(f"Đang lấy review cho bộ lọc: {rating_filter}")
        
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "i[data-hook='review-star-rating']"))
            )
        except Exception as e:
            print(f"Không tìm thấy review cho bộ lọc {rating_filter}: {e}")
            continue
        
        time.sleep(3)
        source = driver.page_source
        soup = bs(source, "html.parser")
        
        review_rating_elem = soup.find("i", {"data-hook": "review-star-rating"})
        if review_rating_elem:
            review_rating_text = review_rating_elem.get_text().strip()
            try:
                review_rating = float(review_rating_text.split(' ')[0])
            except Exception:
                review_rating = None
        else:
            review_rating = None
        
        review_content_elem = soup.find("span", {"data-hook": "review-body"})
        review_content = review_content_elem.get_text().strip() if review_content_elem else ""
        
        results.append({
            "rating_filter": rating_filter,
            "review_rating": review_rating,
            "review_content": review_content
        })
    
    driver.quit()
    df = pd.DataFrame(results)
    return df

# ------------------- GIAO DIỆN CHÍNH VỚI STREAMLIT -------------------

st.sidebar.title("Chọn chức năng")
functionality = st.sidebar.radio("", ("Sentiment Analysis", "Amazon Review Scraper", "Model Evaluation"))

if functionality == "Sentiment Analysis":
    st.title("Sentiment Analysis Interface")
    st.write("Nhập văn bản vào ô dưới đây và nhấn **Predict** để dự đoán cảm xúc của văn bản.")
    
    input_text = st.text_area("Input Text", "I hate you")
    
    if st.button("Predict"):
        if input_text.strip() == "":
            st.warning("Vui lòng nhập văn bản.")
        else:
            predicted_sentiment = predict_sentiment(input_text)
            st.write("Predicted Sentiment:", predicted_sentiment)

elif functionality == "Amazon Review Scraper":
    st.title("Amazon Review Scraper")
    st.write("Nhập thông tin tài khoản Amazon và link review để lấy review và phân loại cảm xúc của review.")
    
    email_amazon = st.text_input("Amazon Email")
    password_amazon = st.text_input("Amazon Password", type="password")
    review_url = st.text_input("Amazon Review URL (Link)")
    
    if st.button("Lấy Reviews"):
        if not email_amazon or not password_amazon or not review_url:
            st.error("Vui lòng nhập đầy đủ thông tin!")
        else:
            with st.spinner("Đang lấy reviews, vui lòng chờ..."):
                try:
                    data = amazon_one_review_per_rating(review_url, email_amazon, password_amazon)
                    st.success("Đã lấy reviews thành công!")
                    if not data.empty:
                        for index, row in data.iterrows():
                            review = row.get("review_content", "")
                            if review.strip() == "":
                                continue
                            review_rating = row.get("review_rating", "N/A")
                            sentiment = predict_sentiment(review)
                            st.markdown(f"**Review Rating:** {review_rating} sao")
                            st.markdown(f"**Review:** {review}")
                            st.markdown(f"*Sentiment:* {sentiment}")
                            st.markdown("---")
                    else:
                        st.write("Không tìm thấy review nào.")
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {e}")

elif functionality == "Model Evaluation":
    st.title("Model Evaluation")
    st.write("Tải file CSV để đánh giá mô hình. File phải có cấu trúc với các cột: sentiment, id, date, query, user, text")
    
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = {"sentiment", "id", "date", "query", "user", "text"}
            if not required_columns.issubset(df.columns):
                st.error("File không có cấu trúc đúng, vui lòng kiểm tra lại.")
            else:
                st.write("File đã được tải thành công. Đang dự đoán cảm xúc cho từng review...")
                df["predicted_sentiment"] = df["text"].apply(lambda x: predict_sentiment(x) if isinstance(x, str) and x.strip() != "" else "Không có review")
                # Map nhãn thực sang dạng chuỗi để so sánh
                def map_true_sentiment(x):
                    if x == 0:
                        return "Tiêu cực"
                    elif x == 4:
                        return "Tích cực"
                    else:
                        return str(x)
                df["sentiment_mapped"] = df["sentiment"].apply(map_true_sentiment)
                
                accuracy = accuracy_score(df["sentiment_mapped"], df["predicted_sentiment"])
                report = classification_report(df["sentiment_mapped"], df["predicted_sentiment"], output_dict=True)
                
                st.write("Accuracy:", accuracy)
                st.write("Classification Report:")
                st.dataframe(pd.DataFrame(report).transpose())
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")
