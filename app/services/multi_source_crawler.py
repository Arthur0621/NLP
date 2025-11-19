"""Multi-source crawler orchestrating feeds -> article insertions."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import feedparser
from dateutil import parser as date_parser
from sqlalchemy.orm import Session

from ..schemas import ArticleCreate, ClassificationResponse
from ..services.classifier import get_classifier_service
from ..services.news_crawler import NewsCrawler
from ..services.feed_sources import FEED_SOURCES
from ..services.summarizer import get_summarizer
from .. import crud

logger = logging.getLogger(__name__)


class MultiSourceCrawler:
    """Pipeline: read RSS -> fetch full content -> save to DB."""

    def __init__(self, db: Session, classify: bool = True):
        self.db = db
        self.classify = classify
        self.crawler = NewsCrawler(delay=0.5)
        self.classifier = get_classifier_service() if classify else None
        self.summarizer = get_summarizer()

    def run(self, limit_per_feed: int = 10) -> List[int]:
        """Fetch latest articles from configured feeds."""

        inserted_ids = []

        for feed in FEED_SOURCES:
            logger.info("Fetching feed %s", feed.name)
            parsed = feedparser.parse(feed.rss_url)
            entries = parsed.entries[:limit_per_feed]

            for entry in entries:
                url = entry.get("link")
                if not url:
                    continue

                # Skip if article already exists
                if crud.get_article_by_url(self.db, url):
                    continue

                content_data = self.crawler.crawl_url(url)
                if not content_data or not content_data.get("content"):
                    continue

                published = self._parse_published(entry)

                summary_text = self._build_summary(entry.get("summary"), content_data.get("content"))

                # Prefer site-specific source from crawler; if it is missing or
                # the generic placeholder, fall back to the feed's source_key.
                raw_source = content_data.get("source")
                source = feed.source_key if not raw_source or raw_source == "generic" else raw_source

                article_in = ArticleCreate(
                    title=content_data.get("title") or entry.get("title", ""),
                    content=content_data.get("content"),
                    source=source,
                    url=url,
                    summary=summary_text,
                    published_at=published,
                    classify=self.classify,
                )

                classification = None
                if self.classify and self.classifier:
                    classification = ClassificationResponse(**self.classifier.classify(article_in.content))

                article = crud.create_article(self.db, article_in, classification)
                inserted_ids.append(article.id)

        return inserted_ids

    @staticmethod
    def _parse_published(entry) -> datetime | None:
        published = entry.get("published") or entry.get("updated")
        if not published:
            return None
        try:
            return date_parser.parse(published)
        except Exception:
            return None

    def _build_summary(self, existing_summary: str | None, content: str | None) -> str | None:
        """Return best-effort summary for article."""

        summary = (existing_summary or "").strip()
        if summary and len(summary.split()) >= 25:
            return summary

        if not content:
            return summary or None

        truncated = " ".join(content.split()[:500])
        try:
            return self.summarizer.summarize(truncated)
        except Exception as exc:  # pragma: no cover
            logger.warning("Summarization failed: %s", exc)
            return summary or truncated[:400]
