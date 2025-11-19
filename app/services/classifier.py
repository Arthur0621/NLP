"""Classifier service that wraps the trained PhoBERT model for reuse."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import json

from ..config import get_settings

settings = get_settings()


class NewsClassifierService:
    """Load and serve the fine-tuned PhoBERT model."""

    def __init__(self, model_path: str | Path, device: str | None = None, top_k: int = 3):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path.as_posix())
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path.as_posix())
        self.model.to(self.device)
        self.model.eval()

        # Load label mapping if available
        label_mapping_file = self.model_path / "label_mapping.json"
        if label_mapping_file.exists():
            with open(label_mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            # Ensure keys are ints
            self.id2label = {int(k): v for k, v in mapping.items()}
        else:
            self.id2label = getattr(self.model.config, "id2label", {})

    def classify(self, text: str, top_k: int | None = None) -> dict:
        """Return predicted topic with confidences."""

        if not text or not text.strip():
            raise ValueError("Input text must not be empty")

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        k = min(top_k or self.top_k, probabilities.shape[-1])
        top_probs, top_indices = torch.topk(probabilities[0], k=k)

        top_predictions = [
            {
                "topic": self.id2label.get(idx.item(), str(idx.item())),
                "confidence": prob.item(),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        predicted_topic = self.id2label.get(predicted_class, str(predicted_class))

        return {
            "predicted_topic": predicted_topic,
            "confidence": confidence,
            "top_predictions": top_predictions,
        }


@lru_cache
def get_classifier_service() -> NewsClassifierService:
    """Return a cached classifier service instance."""

    return NewsClassifierService(
        model_path=settings.model_path,
        top_k=settings.top_k_predictions,
    )
