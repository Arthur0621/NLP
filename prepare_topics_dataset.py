"""Map raw topics to target 10-topic schema for training."""
from __future__ import annotations

import argparse
import os
from collections import Counter

import pandas as pd

from config.topics import TOPIC_MAPPING, TARGET_TOPICS


def normalize_topic(value: str) -> str:
    return value.strip().lower()


def build_reverse_mapping() -> dict[str, str]:
    rev = {}
    for target, raw_list in TOPIC_MAPPING.items():
        for raw in raw_list:
            rev[normalize_topic(raw)] = target
    return rev


def map_topics(df: pd.DataFrame, topic_column: str = "topic") -> pd.DataFrame:
    rev_map = build_reverse_mapping()
    mapped = []
    for topic in df[topic_column].fillna(""):
        mapped.append(rev_map.get(normalize_topic(topic), None))
    df = df.copy()
    df["mapped_topic"] = mapped
    df = df[df["mapped_topic"].notna()]
    df = df[df["mapped_topic"].isin(TARGET_TOPICS)]
    return df


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dataset 10 chủ đề cho huấn luyện")
    parser.add_argument("--input", default="Dataset/processed_news_advanced.csv", help="Đường dẫn file CSV đầu vào")
    parser.add_argument("--output", default="Dataset/training_topics_10.csv", help="File CSV đầu ra")
    parser.add_argument("--min-per-topic", type=int, default=0, help="Giới hạn tối thiểu số mẫu mỗi chủ đề (0 = không lọc)")
    args = parser.parse_args()

    print(f"Đang đọc dữ liệu từ {args.input} ...")
    df = pd.read_csv(args.input)
    if "topic" not in df.columns:
        raise ValueError("File đầu vào phải chứa cột 'topic'")

    df = map_topics(df, topic_column="topic")
    df = df.rename(columns={"mapped_topic": "topic_10"})

    counts = Counter(df["topic_10"])
    print("\nPhân bố sau khi mapping:")
    for topic in TARGET_TOPICS:
        print(f"  - {topic}: {counts.get(topic, 0)}")

    if args.min_per_topic > 0:
        print(f"\nLọc các chủ đề có ít hơn {args.min_per_topic} mẫu")
        df = df[df.groupby("topic_10")["topic_10"].transform("count") >= args.min_per_topic]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nĐã lưu {len(df)} mẫu vào {args.output}")


if __name__ == "__main__":
    main()
