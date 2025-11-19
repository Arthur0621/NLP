"""
Script huấn luyện mô hình PhoBERT cho phân loại tin tức tiếng Việt
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng device: {device}")

class NewsDataset(Dataset):
    """Dataset cho tin tức"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data(data_file, topic_column='topic'):
    """Đọc và chuẩn bị dữ liệu"""
    print(f"Đang đọc dữ liệu từ {data_file}...")
    df = pd.read_csv(data_file)
    
    if topic_column not in df.columns:
        raise ValueError(f"File dữ liệu cần có cột {topic_column}")

    print(f"Tổng số mẫu: {len(df)}")
    print(f"Số chủ đề: {df[topic_column].nunique()}")
    
    # Encode nhãn
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df[topic_column])
    
    # Lưu mapping nhãn
    label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}
    
    return df, label_encoder, label_mapping

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Huấn luyện một epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Chuyển dữ liệu sang device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Thống kê
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1

def evaluate(model, dataloader, device):
    """Đánh giá mô hình"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1, predictions, true_labels

def train_model(
    data_file,
    model_name='vinai/phobert-base',
    epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    max_length=256,
    topic_column='topic',
):
    """Huấn luyện mô hình PhoBERT"""
    
    # Tạo thư mục lưu mô hình
    os.makedirs('models', exist_ok=True)
    
    # Đọc và chuẩn bị dữ liệu
    df, label_encoder, label_mapping = load_and_prepare_data(data_file, topic_column=topic_column)
    
    # Chia train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Load tokenizer và model
    print(f"\nĐang tải mô hình {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_mapping)
    )
    model.to(device)
    
    # Tạo datasets
    train_dataset = NewsDataset(train_df['processed_content'].values, train_df['label'].values, tokenizer, max_length)
    val_dataset = NewsDataset(val_df['processed_content'].values, val_df['label'].values, tokenizer, max_length)
    test_dataset = NewsDataset(test_df['processed_content'].values, test_df['label'].values, tokenizer, max_length)
    
    # Tạo dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Optimizer và scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Huấn luyện
    print("\n" + "="*50)
    print("BẮT ĐẦU HUẤN LUYỆN")
    print("="*50)
    
    best_val_f1 = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Lưu mô hình tốt nhất
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"Lưu mô hình tốt nhất (F1: {val_f1:.4f})...")
            model.save_pretrained('models/best_model')
            tokenizer.save_pretrained('models/best_model')
    
    # Đánh giá trên tập test
    print("\n" + "="*50)
    print("ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*50)
    
    # Load mô hình tốt nhất
    model = AutoModelForSequenceClassification.from_pretrained('models/best_model')
    model.to(device)
    
    test_loss, test_acc, test_f1, predictions, true_labels = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    # Classification report & confusion matrix
    report_text = classification_report(true_labels, predictions, target_names=label_encoder.classes_)
    print("\nClassification Report:")
    print(report_text)

    report_dict = classification_report(true_labels, predictions, target_names=label_encoder.classes_, output_dict=True)
    cm = confusion_matrix(true_labels, predictions)

    os.makedirs('reports', exist_ok=True)
    with open('reports/classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    # Vẽ heatmap confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

    # Lưu label mapping
    with open('models/best_model/label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    
    print("\nHoàn thành huấn luyện!")
    print(f"Mô hình đã được lưu tại: models/best_model/")

if __name__ == "__main__":
    # Cấu hình
    DATA_FILE = "Dataset/training_topics_10.csv"
    MODEL_NAME = "vinai/phobert-base"
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 256
    
    # Huấn luyện
    train_model(
        data_file=DATA_FILE,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        topic_column='topic_10'
    )
