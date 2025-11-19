"""Configuration for target topics and mapping rules."""
from __future__ import annotations

from collections import OrderedDict

TARGET_TOPICS = [
    "Thời sự - Chính trị",
    "Kinh tế - Tài chính",
    "Thế giới",
    "Khoa học - Công nghệ",
    "Kinh doanh & Startup",
    "Thể thao",
    "Giải trí - Văn hóa",
    "Giáo dục",
    "Sức khỏe - Y tế",
    "Đời sống - Du lịch",
]

# Mapping từ tên chủ đề gốc trong dataset sang 10 chủ đề mục tiêu
TOPIC_MAPPING = OrderedDict(
    {
        "Thời sự - Chính trị": [
            "Thời sự",
            "Thời sự - Xã hội",
            "Chính trị",
            "Chính trị - Xã hội",
            "Pháp luật",
            "Hồ sơ - Phân tích",
            "An ninh - Quốc phòng",
            "Bạn đọc",
        ],
        "Kinh tế - Tài chính": [
            "Kinh tế",
            "Kinh tế - Xã hội",
            "Tài chính - Kinh doanh",
            "Tài chính",
            "Ngân hàng",
            "Thị trường",
            "Bất động sản",
            "Vĩ mô",
        ],
        "Thế giới": [
            "Thế giới",
            "Tin thế giới",
            "Quốc tế",
            "Thế giới - Chuyện lạ",
        ],
        "Khoa học - Công nghệ": [
            "Khoa học",
            "Khoa học - Công nghệ",
            "Công nghệ",
            "Số hóa",
            "Khoa học - Đời sống",
        ],
        "Kinh doanh & Startup": [
            "Kinh doanh",
            "Doanh nghiệp",
            "Khởi nghiệp",
            "Startup",
            "Logistics",
        ],
        "Thể thao": [
            "Thể thao",
            "Bóng đá",
            "Thể thao trong nước",
            "Thể thao quốc tế",
            "Soi kèo",
        ],
        "Giải trí - Văn hóa": [
            "Giải trí",
            "Showbiz",
            "Văn hóa",
            "Âm nhạc",
            "Du lịch - Văn hóa",
        ],
        "Giáo dục": [
            "Giáo dục",
            "Hướng nghiệp",
            "Du học",
            "Tuyển sinh",
        ],
        "Sức khỏe - Y tế": [
            "Sức khỏe",
            "Y tế",
            "Dinh dưỡng",
            "COVID-19",
        ],
        "Đời sống - Du lịch": [
            "Đời sống",
            "Du lịch",
            "Nhịp sống",
            "Ẩm thực",
            "Cộng đồng",
        ],
    }
)
