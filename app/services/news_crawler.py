"""Reusable crawler service for Vietnamese news sites."""
from __future__ import annotations

import re
import time
from typing import Optional, Dict, Any

import requests
from bs4 import BeautifulSoup


class NewsCrawler:
    def __init__(self, timeout: int = 10, delay: float = 0.0):
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            }
        )

    # ----- Site specific parsers -------------------------------------------------
    def crawl_vnexpress(self, url: str) -> Optional[Dict[str, Any]]:
        soup = self._get_soup(url)
        if not soup:
            return None

        title_tag = soup.find("h1", class_="title-detail")
        description_tag = soup.find("p", class_="description")
        content_div = soup.find("article", class_="fck_detail")
        paragraphs = content_div.find_all("p", class_="Normal") if content_div else []

        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        full_content = f"{description_tag.get_text(strip=True) if description_tag else ''} {text}".strip()

        return {
            "title": title_tag.get_text(strip=True) if title_tag else "",
            "content": full_content,
            "source": "vnexpress",
            "url": url,
        }

    def crawl_tuoitre(self, url: str) -> Optional[Dict[str, Any]]:
        soup = self._get_soup(url)
        if not soup:
            return None

        title_tag = soup.find("h1", class_="detail-title")
        desc_tag = soup.find("h2", class_="detail-sapo")
        content_div = soup.find("div", id="main-detail-body")
        paragraphs = content_div.find_all("p") if content_div else []

        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        full_content = f"{desc_tag.get_text(strip=True) if desc_tag else ''} {text}".strip()

        return {
            "title": title_tag.get_text(strip=True) if title_tag else "",
            "content": full_content,
            "source": "tuoitre",
            "url": url,
        }

    def crawl_thanhnien(self, url: str) -> Optional[Dict[str, Any]]:
        soup = self._get_soup(url)
        if not soup:
            return None

        title_tag = soup.find("h1", class_="details__headline")
        desc_tag = soup.find("div", class_="details__summary")
        content_div = soup.find("div", id="contentDetail")
        paragraphs = content_div.find_all("p") if content_div else []

        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        full_content = f"{desc_tag.get_text(strip=True) if desc_tag else ''} {text}".strip()

        return {
            "title": title_tag.get_text(strip=True) if title_tag else "",
            "content": full_content,
            "source": "thanhnien",
            "url": url,
        }

    def crawl_dantri(self, url: str) -> Optional[Dict[str, Any]]:
        soup = self._get_soup(url)
        if not soup:
            return None

        title_tag = soup.find("h1", class_="dt-news__title")
        desc_tag = soup.find("h2", class_="dt-news__sapo")
        content_div = soup.find("div", class_="dt-news__content")
        paragraphs = content_div.find_all("p") if content_div else []

        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        full_content = f"{desc_tag.get_text(strip=True) if desc_tag else ''} {text}".strip()

        return {
            "title": title_tag.get_text(strip=True) if title_tag else "",
            "content": full_content,
            "source": "dantri",
            "url": url,
        }

    # ----- Generic fallback ------------------------------------------------------
    def crawl_generic(self, url: str) -> Optional[Dict[str, Any]]:
        soup = self._get_soup(url)
        if not soup:
            return None

        title = self._extract_title(soup)
        content = self._extract_content(soup)

        return {
            "title": title,
            "content": content,
            "source": "generic",
            "url": url,
        }

    def crawl_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Detect site by URL and crawl full content."""

        try:
            if "vnexpress.net" in url:
                result = self.crawl_vnexpress(url)
            elif "tuoitre.vn" in url:
                result = self.crawl_tuoitre(url)
            elif "thanhnien.vn" in url:
                result = self.crawl_thanhnien(url)
            elif "dantri.com.vn" in url:
                result = self.crawl_dantri(url)
            else:
                result = self.crawl_generic(url)

            if self.delay:
                time.sleep(self.delay)

            return result
        except Exception:
            return None

    # ----- Internal helpers ------------------------------------------------------
    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.content, "html.parser")
        except Exception:
            return None

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        candidates = [
            soup.find("h1"),
            soup.find("meta", attrs={"property": "og:title"}),
            soup.find("title"),
        ]
        for tag in candidates:
            if not tag:
                continue
            if tag.name == "meta":
                value = tag.get("content", "").strip()
            else:
                value = tag.get_text(strip=True)
            if value:
                return value
        return ""

    @staticmethod
    def _extract_content(soup: BeautifulSoup) -> str:
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
            if paragraphs:
                return " ".join(p.get_text(strip=True) for p in paragraphs)

        pattern = re.compile(r"(content|article|body|detail)", re.I)
        divs = soup.find_all("div", class_=pattern)
        for div in divs:
            paragraphs = div.find_all("p")
            if len(paragraphs) >= 3:
                return " ".join(p.get_text(strip=True) for p in paragraphs)

        fallback = soup.find_all("p")
        return " ".join(p.get_text(strip=True) for p in fallback[:20])
