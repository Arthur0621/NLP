"""
Script tiền xử lý dữ liệu nâng cao cho tin tức tiếng Việt
Dựa trên quy trình: lowercase -> remove URLs -> remove punctuation -> 
tokenize -> remove stopwords -> stemming
"""
import pandas as pd
import numpy as np
import json
import re
import string
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os

class VietnameseTextPreprocessor:
    """Class xử lý văn bản tiếng Việt"""
    
    def __init__(self, stopwords_file=None):
        """
        Khởi tạo preprocessor
        
        Args:
            stopwords_file: Đường dẫn file stopwords tiếng Việt
        """
        self.stopwords = set()
        
        # Load stopwords nếu có
        if stopwords_file and os.path.exists(stopwords_file):
            self.load_stopwords(stopwords_file)
        else:
            # Sử dụng stopwords mặc định
            self.stopwords = self.get_default_stopwords()
        
        # Khởi tạo stemmer
        self.stemmer = PorterStemmer()
        
        # Pattern để tìm URLs
        self.url_pattern = r'(http\S+|https\S+|www\S+)'
    
    def load_stopwords(self, filepath):
        """Load stopwords từ file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            tmp = line.rstrip()
            tmp_no_digits = ''.join([i for i in tmp if not i.isdigit()])
            if tmp_no_digits.strip():
                self.stopwords.add(tmp_no_digits)
        
        print(f"✓ Đã load {len(self.stopwords)} stopwords")
    
    def get_default_stopwords(self):
        """Trả về danh sách stopwords tiếng Việt mặc định"""
        return {
            'và', 'của', 'có', 'trong', 'là', 'được', 'cho', 'với', 'này', 'các',
            'để', 'một', 'người', 'không', 'từ', 'đã', 'những', 'theo', 'như', 'khi',
            'về', 'tại', 'hay', 'đến', 'bị', 'do', 'nên', 'vì', 'nếu', 'mà',
            'thì', 'đang', 'sẽ', 'cũng', 'rất', 'nhiều', 'lại', 'ra', 'vào', 'còn',
            'nữa', 'đây', 'đó', 'nó', 'họ', 'chúng', 'tôi', 'bạn', 'anh', 'chị',
            'em', 'ông', 'bà', 'cô', 'chú', 'thầy', 'cậu', 'mình', 'ta'
        }
    
    def lowercase(self, text):
        """Chuyển văn bản về chữ thường"""
        return text.lower() if isinstance(text, str) else ""
    
    def extract_urls(self, text):
        """Trích xuất URLs từ văn bản"""
        if not isinstance(text, str):
            return []
        urls = re.findall(self.url_pattern, text)
        return urls
    
    def remove_urls(self, text):
        """Loại bỏ URLs khỏi văn bản"""
        if not isinstance(text, str):
            return ""
        return re.sub(self.url_pattern, '', text)
    
    def remove_punctuation(self, text):
        """Loại bỏ dấu câu"""
        if not isinstance(text, str):
            return ""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_special_chars(self, text):
        """Loại bỏ ký tự đặc biệt, giữ lại chữ cái và số"""
        if not isinstance(text, str):
            return ""
        # Giữ lại chữ cái tiếng Việt, số và khoảng trắng
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text):
        """Tokenize văn bản thành các từ"""
        if not isinstance(text, str) or not text.strip():
            return []
        try:
            tokens = word_tokenize(text)
            return tokens
        except:
            # Fallback: tách theo khoảng trắng
            return text.split()
    
    def remove_stopwords(self, tokens):
        """Loại bỏ stopwords"""
        if not isinstance(tokens, list):
            return []
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def stem_tokens(self, tokens):
        """Stemming các tokens"""
        if not isinstance(tokens, list):
            return []
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text, extract_urls_flag=False):
        """
        Xử lý toàn bộ pipeline cho một văn bản
        
        Args:
            text: Văn bản đầu vào
            extract_urls_flag: Có trích xuất URLs không
            
        Returns:
            dict: Kết quả xử lý
        """
        result = {
            'original': text,
            'urls': [],
            'lowercase': '',
            'no_urls': '',
            'no_punctuation': '',
            'tokenized': [],
            'no_stopwords': [],
            'stemmed': [],
            'processed': ''
        }
        
        # Bước 1: Lowercase
        result['lowercase'] = self.lowercase(text)
        
        # Bước 2: Extract URLs (nếu cần)
        if extract_urls_flag:
            result['urls'] = self.extract_urls(result['lowercase'])
        
        # Bước 3: Remove URLs
        result['no_urls'] = self.remove_urls(result['lowercase'])
        
        # Bước 4: Remove punctuation và special chars
        result['no_punctuation'] = self.remove_special_chars(result['no_urls'])
        
        # Bước 5: Tokenize
        result['tokenized'] = self.tokenize(result['no_punctuation'])
        
        # Bước 6: Remove stopwords
        result['no_stopwords'] = self.remove_stopwords(result['tokenized'])
        
        # Bước 7: Stemming
        result['stemmed'] = self.stem_tokens(result['no_stopwords'])
        
        # Bước 8: Join lại thành văn bản
        result['processed'] = ' '.join(result['stemmed'])
        
        return result

def preprocess_dataset(
    input_file='Dataset/news_dataset.json',
    output_file='Dataset/processed_news_advanced.csv',
    stopwords_file=None,
    max_samples=50000
):
    """
    Tiền xử lý toàn bộ dataset
    
    Args:
        input_file: File JSON đầu vào
        output_file: File CSV đầu ra
        stopwords_file: File stopwords tiếng Việt
        max_samples: Số lượng mẫu tối đa
    """
    print("="*70)
    print("TIỀN XỬ LÝ DỮ LIỆU TIN TỨC TIẾNG VIỆT")
    print("="*70)
    
    # Bước 1: Đọc dữ liệu
    print(f"\n[1/4] Đang đọc dữ liệu từ {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"✓ Tổng số bản ghi: {len(json_data)}")
    
    # Giới hạn số lượng
    if max_samples and len(json_data) > max_samples:
        print(f"  Giới hạn xuống {max_samples} mẫu...")
        json_data = json_data[:max_samples]
    
    # Bước 2: Khởi tạo preprocessor
    print(f"\n[2/4] Khởi tạo preprocessor...")
    preprocessor = VietnameseTextPreprocessor(stopwords_file)
    
    # Bước 3: Xử lý dữ liệu
    print(f"\n[3/4] Đang xử lý dữ liệu...")
    
    processed_data = []
    skip_count = 0
    
    for item in tqdm(json_data, desc="Xử lý"):
        try:
            # Lấy thông tin
            id_ = item.get('id', '')
            author = item.get('author', '')
            source = item.get('source', '')
            title = item.get('title', '')
            content = item.get('content', '')
            topic = item.get('topic', '')
            url = item.get('url', '')
            
            # Bỏ qua nếu thiếu thông tin quan trọng
            if not title or not content or not topic:
                skip_count += 1
                continue
            
            # Kết hợp title và content
            full_text = f"{title}. {content}"
            
            # Xử lý văn bản
            result = preprocessor.preprocess_text(full_text, extract_urls_flag=True)
            
            # Bỏ qua nếu văn bản quá ngắn sau xử lý
            if len(result['processed']) < 50:
                skip_count += 1
                continue
            
            # Lưu kết quả
            processed_data.append({
                'id': id_,
                'author': author,
                'source': source,
                'title': title,
                'topic': topic,
                'url': url,
                'original_content': content,
                'lowercase_content': result['lowercase'],
                'tokenized_content': ' '.join(result['tokenized']),
                'no_stopwords_content': ' '.join(result['no_stopwords']),
                'stemmed_content': ' '.join(result['stemmed']),
                'processed_content': result['processed'],
                'urls_extracted': ', '.join(result['urls']) if result['urls'] else ''
            })
            
        except Exception as e:
            skip_count += 1
            continue
    
    print(f"\n✓ Đã xử lý: {len(processed_data)} bản ghi")
    print(f"✓ Đã bỏ qua: {skip_count} bản ghi")
    
    # Bước 4: Tạo DataFrame và lưu
    print(f"\n[4/4] Đang tạo DataFrame và lưu kết quả...")
    df = pd.DataFrame(processed_data)
    
    # Lọc các chủ đề có ít mẫu
    topic_counts = df['topic'].value_counts()
    valid_topics = topic_counts[topic_counts >= 10].index
    df = df[df['topic'].isin(valid_topics)]
    
    # Loại bỏ duplicate
    df = df.drop_duplicates(subset=['processed_content'])
    
    print(f"\n✓ Số lượng bản ghi cuối cùng: {len(df)}")
    print(f"✓ Số lượng chủ đề: {df['topic'].nunique()}")
    
    # Thống kê chủ đề
    print("\nPhân bố chủ đề (Top 10):")
    for topic, count in df['topic'].value_counts().head(10).items():
        print(f"  - {topic}: {count}")
    
    # Lưu file
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ Đã lưu dữ liệu vào: {output_file}")
    
    # Thống kê
    print("\n" + "="*70)
    print("THỐNG KÊ DỮ LIỆU")
    print("="*70)
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Số chủ đề: {df['topic'].nunique()}")
    print(f"Độ dài trung bình (processed): {df['processed_content'].str.len().mean():.0f} ký tự")
    print(f"Số từ trung bình: {df['processed_content'].str.split().str.len().mean():.0f} từ")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tiền xử lý dữ liệu tin tức tiếng Việt (nâng cao)')
    parser.add_argument('--input', default='Dataset/news_dataset.json', help='File JSON đầu vào')
    parser.add_argument('--output', default='Dataset/processed_news_advanced.csv', help='File CSV đầu ra')
    parser.add_argument('--stopwords', default=None, help='File stopwords tiếng Việt')
    parser.add_argument('--max-samples', type=int, default=50000, help='Số lượng mẫu tối đa')
    
    args = parser.parse_args()
    
    # Xử lý dữ liệu
    df = preprocess_dataset(
        input_file=args.input,
        output_file=args.output,
        stopwords_file=args.stopwords,
        max_samples=args.max_samples
    )
