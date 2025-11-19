"""SQLAlchemy models for news articles and topics."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .database import Base


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(512), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(128), nullable=False)
    url = Column(String(512), unique=True, nullable=False)
    summary = Column(Text, nullable=True)
    topic = Column(String(128), nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    published_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    predictions = relationship("ArticlePrediction", back_populates="article", cascade="all, delete-orphan")


class ArticlePrediction(Base):
    __tablename__ = "article_predictions"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    topic = Column(String(128), nullable=False)
    confidence = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    extra = Column(JSONB, nullable=True)

    article = relationship("Article", back_populates="predictions")
