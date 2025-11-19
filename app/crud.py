"""Database CRUD helpers."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import select, func, or_, distinct

from . import models, schemas


def create_article(
    db: Session,
    article_in: schemas.ArticleCreate,
    classification: Optional[schemas.ClassificationResponse] = None,
) -> models.Article:
    article = models.Article(
        title=article_in.title,
        content=article_in.content,
        source=article_in.source,
        url=str(article_in.url),
        summary=article_in.summary,
        published_at=article_in.published_at,
    )

    if classification:
        article.topic = classification.predicted_topic
        article.confidence = classification.confidence

    db.add(article)
    db.flush()  # to obtain article.id

    if classification:
        predictions = []
        for idx, pred in enumerate(classification.top_predictions, start=1):
            predictions.append(
                models.ArticlePrediction(
                    article_id=article.id,
                    topic=pred.topic,
                    confidence=pred.confidence,
                    rank=idx,
                )
            )
        db.add_all(predictions)

    db.commit()
    db.refresh(article)
    return article


def list_articles(
    db: Session,
    *,
    topic: Optional[str] = None,
    source: Optional[str] = None,
    query_text: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[models.Article], int]:
    query = select(models.Article)
    count_query = select(func.count()).select_from(models.Article)

    if topic:
        query = query.where(models.Article.topic == topic)
        count_query = count_query.where(models.Article.topic == topic)

    if source:
        query = query.where(models.Article.source == source)
        count_query = count_query.where(models.Article.source == source)

    if query_text:
        pattern = f"%{query_text}%"
        condition = or_(
            models.Article.title.ilike(pattern),
            models.Article.content.ilike(pattern),
        )
        query = query.where(condition)
        count_query = count_query.where(condition)

    query = query.order_by(models.Article.published_at.desc().nullslast(), models.Article.created_at.desc())
    items = db.execute(query.offset(offset).limit(limit)).scalars().unique().all()
    total = db.execute(count_query).scalar_one()
    return items, total


def get_article(db: Session, article_id: int) -> Optional[models.Article]:
    return db.get(models.Article, article_id)


def get_article_by_url(db: Session, url: str) -> Optional[models.Article]:
    stmt = select(models.Article).where(models.Article.url == url)
    return db.execute(stmt).scalars().first()


def get_stats(db: Session, hours: int = 24, limit: int = 10) -> dict:
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    topic_stmt = (
        select(models.Article.topic, func.count())
        .where(models.Article.created_at >= cutoff, models.Article.topic.is_not(None))
        .group_by(models.Article.topic)
        .order_by(func.count().desc())
        .limit(limit)
    )

    source_stmt = (
        select(models.Article.source, func.count())
        .where(models.Article.created_at >= cutoff)
        .group_by(models.Article.source)
        .order_by(func.count().desc())
        .limit(limit)
    )

    topics = [
        {"topic": row[0], "count": row[1]}
        for row in db.execute(topic_stmt)
        if row[0]
    ]
    sources = [
        {"source": row[0], "count": row[1]}
        for row in db.execute(source_stmt)
        if row[0]
    ]

    source_topic_counts = (
        select(
            models.Article.source.label("source"),
            models.Article.topic.label("topic"),
            func.count().label("count"),
        )
        .where(models.Article.created_at >= cutoff, models.Article.topic.is_not(None))
        .group_by(models.Article.source, models.Article.topic)
    ).subquery()

    max_counts = (
        select(
            source_topic_counts.c.source,
            func.max(source_topic_counts.c.count).label("max_count"),
        )
        .group_by(source_topic_counts.c.source)
    ).subquery()

    top_source_topics_stmt = (
        select(
            source_topic_counts.c.source,
            source_topic_counts.c.topic,
            source_topic_counts.c.count,
        )
        .join(
            max_counts,
            (source_topic_counts.c.source == max_counts.c.source)
            & (source_topic_counts.c.count == max_counts.c.max_count),
        )
        .order_by(source_topic_counts.c.count.desc())
        .limit(limit)
    )

    source_topics = [
        {"source": row[0], "topic": row[1], "count": row[2]}
        for row in db.execute(top_source_topics_stmt)
        if row[0] and row[1]
    ]

    return {"topics": topics, "sources": sources, "source_topics": source_topics}


def get_daily_topic_stats(db: Session, days: int = 7) -> list[dict]:
    """Return the top topic for each day in the last `days` days.

    The result is a list of dicts with keys: date, topic, count.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    day_topic_counts = (
        select(
            func.date(models.Article.created_at).label("day"),
            models.Article.topic.label("topic"),
            func.count().label("count"),
        )
        .where(models.Article.created_at >= cutoff, models.Article.topic.is_not(None))
        .group_by("day", models.Article.topic)
    ).subquery()

    max_counts = (
        select(
            day_topic_counts.c.day,
            func.max(day_topic_counts.c.count).label("max_count"),
        )
        .group_by(day_topic_counts.c.day)
    ).subquery()

    top_stmt = (
        select(
            day_topic_counts.c.day,
            day_topic_counts.c.topic,
            day_topic_counts.c.count,
        )
        .join(
            max_counts,
            (day_topic_counts.c.day == max_counts.c.day)
            & (day_topic_counts.c.count == max_counts.c.max_count),
        )
        .order_by(day_topic_counts.c.day.desc())
    )

    return [
        {"date": row[0], "topic": row[1], "count": row[2]}
        for row in db.execute(top_stmt)
        if row[0] and row[1]
    ]


def list_topics(db: Session, limit: int = 20) -> list[str]:
    stmt = (
        select(distinct(models.Article.topic))
        .where(models.Article.topic.is_not(None))
        .order_by(models.Article.topic)
        .limit(limit)
    )
    return [row[0] for row in db.execute(stmt) if row[0]]


def get_articles_by_topic(db: Session, topic: str, limit: int = 10) -> list[models.Article]:
    stmt = (
        select(models.Article)
        .where(models.Article.topic == topic)
        .order_by(models.Article.published_at.desc().nullslast(), models.Article.created_at.desc())
        .limit(limit)
    )
    return db.execute(stmt).scalars().unique().all()
