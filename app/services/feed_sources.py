"""Feed source configuration for the crawler job."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedSource:
    name: str
    rss_url: str
    source_key: str


FEED_SOURCES: list[FeedSource] = [
    FeedSource("VnExpress - Thời sự", "https://vnexpress.net/rss/thoi-su.rss", "vnexpress"),
    FeedSource("VnExpress - Thế giới", "https://vnexpress.net/rss/the-gioi.rss", "vnexpress"),
    FeedSource("Tuổi Trẻ - Tin mới", "https://tuoitre.vn/rss/tin-moi-nhat.rss", "tuoitre"),
    FeedSource("Thanh Niên - Trang chủ", "https://thanhnien.vn/rss/home.rss", "thanhnien"),
    FeedSource("Dân Trí - Trang chủ", "https://dantri.com.vn/rss/home.rss", "dantri"),
    FeedSource("VOV - Tin mới 24h", "https://vov.vn/rss/tin-moi-nhat-24h.rss", "vov"),
    FeedSource("VietnamNet - Tin mới", "https://vietnamnet.vn/rss/tin-moi-nhat.rss", "vietnamnet"),
    FeedSource("Zing News - Tin mới", "https://znews.vn/rss/tin-moi.rss", "znews"),
    FeedSource("Lao Động", "https://laodong.vn/rss", "laodong"),
    FeedSource("VietnamPlus - Việt Nam", "https://www.vietnamplus.vn/rss/timeline/viet-nam.rss", "vietnamplus"),
]
