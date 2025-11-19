"""
Script dự đoán chủ đề tin tức sử dụng mô hình đã huấn luyện
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

class NewsClassifier:
    """Class để phân loại tin tức"""
    
    def __init__(self, model_path='models/best_model'):
        """
        Khởi tạo classifier
        
        Args:
            model_path: Đường dẫn đến thư mục chứa mô hình
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng device: {self.device}")
        
        # Load model và tokenizer
        print(f"Đang tải mô hình từ {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        with open(f'{model_path}/label_mapping.json', 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
        
        # Chuyển key từ string sang int
        self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
        
        print("Mô hình đã sẵn sàng!")
    
    def predict(self, text, max_length=256):
        """
        Dự đoán chủ đề của văn bản
        
        Args:
            text: Văn bản cần phân loại
            max_length: Độ dài tối đa của văn bản
            
        Returns:
            dict: Kết quả dự đoán bao gồm nhãn và xác suất
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Lấy kết quả
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Lấy top 3 dự đoán
        top_probs, top_indices = torch.topk(probabilities[0], k=min(3, len(self.label_mapping)))
        top_predictions = [
            {
                'topic': self.label_mapping[idx.item()],
                'confidence': prob.item()
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return {
            'predicted_topic': self.label_mapping[predicted_class],
            'confidence': confidence,
            'top_predictions': top_predictions
        }
    
    def predict_batch(self, texts, max_length=256, batch_size=16):
        """
        Dự đoán cho nhiều văn bản
        
        Args:
            texts: List các văn bản cần phân loại
            max_length: Độ dài tối đa của văn bản
            batch_size: Kích thước batch
            
        Returns:
            list: Danh sách kết quả dự đoán
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Xử lý kết quả
            for j in range(len(batch_texts)):
                predicted_class = torch.argmax(probabilities[j]).item()
                confidence = probabilities[j][predicted_class].item()
                
                results.append({
                    'text': batch_texts[j][:100] + '...',  # Hiển thị 100 ký tự đầu
                    'predicted_topic': self.label_mapping[predicted_class],
                    'confidence': confidence
                })
        
        return results

def main():
    """Hàm main để test"""
    # Khởi tạo classifier
    classifier = NewsClassifier('models/best_model')
    
    # Test với một số ví dụ
    test_texts = [
        "Đội tuyển Việt Nam giành chiến thắng 3-0 trước Thái Lan trong trận chung kết AFF Cup",
        "Chính phủ công bố gói hỗ trợ kinh tế 100 nghìn tỷ đồng cho doanh nghiệp",
        "Ca sĩ Sơn Tùng M-TP phát hành MV mới đạt 10 triệu view sau 24 giờ",
        "Giá vàng trong nước tăng mạnh lên mức cao nhất trong 5 năm",
        "Học sinh lớp 10 giành huy chương vàng Olympic Toán học quốc tế"
    ]
    
    print("\n" + "="*70)
    print("DEMO DỰ ĐOÁN CHỦ ĐỀ TIN TỨC")
    print("="*70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Văn bản: {text}")
        result = classifier.predict(text)
        print(f"   Chủ đề dự đoán: {result['predicted_topic']}")
        print(f"   Độ tin cậy: {result['confidence']:.2%}")
        print(f"   Top 3 dự đoán:")
        for pred in result['top_predictions']:
            print(f"      - {pred['topic']}: {pred['confidence']:.2%}")

if __name__ == "__main__":
    main()
