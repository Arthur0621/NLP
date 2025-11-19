"""Summarizer service using Vietnamese T5 model."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ..config import get_settings


class NewsSummarizer:
    def __init__(self, model_name: Optional[str] = None, max_length: int = 150):
        settings = get_settings()
        self.model_name = model_name or getattr(settings, "summary_model", "VietAI/vit5-base-vietnews-summarization")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.max_length = max_length

    def summarize(self, text: str, min_length: int = 50) -> str:
        if not text or len(text.split()) < 30:
            return text

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=self.max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


@lru_cache
def get_summarizer() -> NewsSummarizer:
    return NewsSummarizer()
