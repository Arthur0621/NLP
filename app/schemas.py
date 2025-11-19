"""Pydantic schemas for FastAPI endpoints."""
from __future__ import annotations

from datetime import datetime, date
from typing import List, Optional

from pydantic import BaseModel, HttpUrl, Field, ConfigDict


class PredictionBase(BaseModel):
    topic: str
    confidence: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)


class PredictionRead(PredictionBase):
    id: int
    model_config = ConfigDict(from_attributes=True)


class ArticleBase(BaseModel):
    title: str
    content: str
    source: str
    url: HttpUrl
    summary: Optional[str] = None
    published_at: Optional[datetime] = None


class ArticleCreate(ArticleBase):
    classify: bool = True


class ArticleRead(ArticleBase):
    id: int
    topic: Optional[str] = None
    confidence: Optional[float] = None
    created_at: datetime
    predictions: List[PredictionRead] = []
    model_config = ConfigDict(from_attributes=True)


class ArticleListResponse(BaseModel):
    items: List[ArticleRead]
    total: int


class ArticleQueryParams(BaseModel):
    topic: Optional[str] = None
    source: Optional[str] = None
    query: Optional[str] = None
    limit: int = 20
    offset: int = 0


class ClassificationRequest(BaseModel):
    text: str
    top_k: Optional[int] = None


class PredictionItem(BaseModel):
    topic: str
    confidence: float


class ClassificationResponse(BaseModel):
    predicted_topic: str
    confidence: float
    top_predictions: List[PredictionItem]


class TopicStat(BaseModel):
    topic: str
    count: int


class SourceStat(BaseModel):
    source: str
    count: int


class SourceTopTopic(BaseModel):
    source: str
    topic: str
    count: int


class StatsResponse(BaseModel):
    topics: List[TopicStat]
    sources: List[SourceStat]
    source_topics: List[SourceTopTopic] = []


class DailyTopicStat(BaseModel):
    date: date
    topic: str
    count: int


class DailyStatsResponse(BaseModel):
    days: List[DailyTopicStat]


class TopicArticle(BaseModel):
    id: int
    title: str
    summary: Optional[str] = None
    snippet: Optional[str] = None
    url: HttpUrl
    source: str
    published_at: Optional[datetime] = None
    topic: Optional[str] = None

    class Config:
        orm_mode = True


class TopicArticlesResponse(BaseModel):
    topic: str
    items: List[TopicArticle]
    count: int
