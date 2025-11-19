"""FastAPI entrypoint for news classification platform."""
from __future__ import annotations

import logging
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .config import get_settings
from .database import Base, engine, get_db, SessionLocal
from .schemas import (
    ArticleCreate,
    ArticleListResponse,
    ArticleRead,
    ClassificationRequest,
    ClassificationResponse,
    StatsResponse,
    TopicArticlesResponse,
    DailyStatsResponse,
)
from .services.classifier import get_classifier_service
from .services.multi_source_crawler import MultiSourceCrawler
from . import crud

settings = get_settings()
logger = logging.getLogger(__name__)

scheduler: Optional[BackgroundScheduler] = None

app = FastAPI(title="Vietnamese News Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    # preload classifier
    get_classifier_service()
    _start_auto_crawl_scheduler()


@app.on_event("shutdown")
def shutdown_event():
    global scheduler
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        scheduler = None


@app.post("/classify", response_model=ClassificationResponse)
def classify_text(
    payload: ClassificationRequest,
    classifier=Depends(get_classifier_service),
):
    result = classifier.classify(payload.text, payload.top_k)
    return ClassificationResponse(**result)


@app.get("/news", response_model=ArticleListResponse)
def list_news(
    topic: Optional[str] = None,
    source: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    items, total = crud.list_articles(
        db,
        topic=topic,
        source=source,
        query_text=query,
        limit=limit,
        offset=offset,
    )
    return ArticleListResponse(items=items, total=total)


@app.post("/news", response_model=ArticleRead)
def create_news(
    article_in: ArticleCreate,
    db: Session = Depends(get_db),
    classifier=Depends(get_classifier_service),
):
    classification = None
    if article_in.classify:
        classification = classifier.classify(article_in.content)
    article = crud.create_article(db, article_in, classification)
    return article


@app.get("/news/{article_id}", response_model=ArticleRead)
def get_news(article_id: int, db: Session = Depends(get_db)):
    article = crud.get_article(db, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@app.get("/stats", response_model=StatsResponse)
def get_stats(
    hours: int = 24,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    stats = crud.get_stats(db, hours=hours, limit=limit)
    return StatsResponse(**stats)


@app.get("/stats/daily-topics", response_model=DailyStatsResponse)
def get_daily_topics(days: int = 7, db: Session = Depends(get_db)):
    data = crud.get_daily_topic_stats(db, days=days)
    return DailyStatsResponse(days=data)


@app.post("/crawl/feeds")
def crawl_feeds(limit_per_feed: int = 50, classify: bool = True, db: Session = Depends(get_db)):
    crawler = MultiSourceCrawler(db, classify=classify)
    inserted = crawler.run(limit_per_feed=limit_per_feed)
    return {"inserted": len(inserted), "ids": inserted}


@app.get("/topics")
def list_topics(limit: int = 20, db: Session = Depends(get_db)):
    topics = crud.list_topics(db, limit=limit)
    return {"topics": topics}


@app.get("/topics/{topic}", response_model=TopicArticlesResponse)
def topic_articles(topic: str, limit: int = 10, db: Session = Depends(get_db)):
    items = crud.get_articles_by_topic(db, topic=topic, limit=limit)
    response_items = [
        {
            "id": article.id,
            "title": article.title,
            "summary": article.summary,
            "snippet": (article.content[:250] + "...") if article.content else None,
            "url": article.url,
            "source": article.source,
            "published_at": article.published_at,
            "topic": article.topic,
        }
        for article in items
    ]
    return TopicArticlesResponse(topic=topic, items=response_items, count=len(response_items))


def _start_auto_crawl_scheduler():
    global scheduler
    if not settings.auto_crawl_enabled:
        logger.info("Auto crawl scheduler disabled via settings")
        return

    if scheduler and scheduler.running:
        logger.info("Auto crawl scheduler already running")
        return

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _auto_crawl_job,
        trigger=IntervalTrigger(minutes=settings.auto_crawl_interval_minutes),
        id="auto-crawl",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Auto crawl scheduler started: every %s minutes (limit %s per feed)",
        settings.auto_crawl_interval_minutes,
        settings.auto_crawl_limit_per_feed,
    )


def _auto_crawl_job():
    logger.info("Auto crawl job triggered")
    db = SessionLocal()
    try:
        crawler = MultiSourceCrawler(db, classify=True)
        inserted = crawler.run(limit_per_feed=settings.auto_crawl_limit_per_feed)
        logger.info("Auto crawl inserted %s articles", len(inserted))
    except Exception as exc:  # pragma: no cover
        logger.exception("Auto crawl failed: %s", exc)
    finally:
        db.close()
