  const newsPrev = document.getElementById("news-prev");
  const newsNext = document.getElementById("news-next");

if (newsPrev) {
  newsPrev.addEventListener("click", () => {
    if (newsPage > 1) {
      newsPage -= 1;
      loadNews();
    }
  });
}

if (newsNext) {
  newsNext.addEventListener("click", () => {
    newsPage += 1;
    loadNews();
  });
}

const API_BASE = "http://localhost:8000";

// Utility to strip HTML tags from content before display
const cleanHtml = (html) => {
  if (!html) return "";
  const tmp = document.createElement("div");
  tmp.innerHTML = html;
  return tmp.textContent || tmp.innerText || "";
};

// Map internal source keys to friendly display names
const formatSource = (src) => {
  if (!src) return "";
  const map = {
    vnexpress: "VnExpress",
    tuoitre: "Tuổi Trẻ",
    thanhnien: "Thanh Niên",
    dantri: "Dân trí",
    vov: "VOV",
    vietnamnet: "VietnamNet",
    znews: "Zing News",
    laodong: "Lao Động",
    vietnamplus: "VietnamPlus",
    generic: "Khác",
  };
  return map[src] || src;
};

// Pagination state for news list
let newsPage = 1;
const NEWS_PAGE_SIZE = 10;
let newsTotal = 0;

// Chart instances for stats
let topicsChart = null;
let sourcesChart = null;

const classifyText = async () => {
  const text = document.getElementById("classify-text").value.trim();
  const resultBox = document.getElementById("classify-result");

  if (!text) {
    resultBox.textContent = "Vui lòng nhập nội dung.";
    return;
  }

  resultBox.textContent = "Đang phân loại...";
  try {
    const res = await fetch(`${API_BASE}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) throw new Error("API trả lỗi");

    const data = await res.json();
    const lines = [
      `Chủ đề: ${data.predicted_topic} (độ tin cậy ${(data.confidence * 100).toFixed(2)}%)`,
      "Top dự đoán:",
      ...data.top_predictions.map(
        (p, idx) => `${idx + 1}. ${p.topic} (${(p.confidence * 100).toFixed(1)}%)`
      ),
    ];
    resultBox.innerHTML = lines.join("<br>");
  } catch (err) {
    resultBox.textContent = `Lỗi: ${err.message}`;
  }
};

const renderTopicArticle = (item) => {
  const card = document.createElement("div");
  card.className = "topic-card-item";
  const summaryText = cleanHtml(item.summary || item.snippet || "");
  card.innerHTML = `
    <div class="topic-card-thumb">
      <span>${item.topic || "Tin tức"}</span>
    </div>
    <div class="topic-card-body">
      <h4>${item.title}</h4>
      <p class="meta">Nguồn: ${formatSource(item.source)}${item.published_at ? ` · ${new Date(item.published_at).toLocaleString()}` : ""}</p>
      <p>${summaryText.slice(0, 140)}...</p>
      <a href="${item.url}" target="_blank">Đọc chi tiết →</a>
    </div>
  `;
  return card;
};

const loadTopicArticles = async () => {
  const select = document.getElementById("topic-select");
  const topic = select.value;
  const container = document.getElementById("topic-articles");
  if (!topic) {
    container.innerHTML = '<p class="muted">Chọn một chủ đề để xem bài viết.</p>';
    return;
  }

  container.innerHTML = "Đang tải...";
  try {
    const res = await fetch(`${API_BASE}/topics/${encodeURIComponent(topic)}?limit=6`);
    if (!res.ok) throw new Error("Không lấy được bài viết");
    const data = await res.json();
    if (!data.items.length) {
      container.innerHTML = "<p class=\"muted\">Chưa có bài viết cho chủ đề này.</p>";
      return;
    }
    container.innerHTML = "";
    data.items.forEach((item) => container.appendChild(renderTopicArticle(item)));
  } catch (err) {
    container.innerHTML = `Lỗi: ${err.message}`;
  }
};

const loadTopics = async () => {
  const select = document.getElementById("topic-select");
  select.innerHTML = '<option value="" disabled selected>Đang tải...</option>';
  try {
    const res = await fetch(`${API_BASE}/topics?limit=20`);
    if (!res.ok) throw new Error("Không tải được chủ đề");
    const data = await res.json();
    select.innerHTML = "";
    select.appendChild(new Option("Chọn chủ đề", "", true, false));
    data.topics.forEach((topic) => {
      select.appendChild(new Option(topic, topic));
    });
  } catch (err) {
    select.innerHTML = `<option>${err.message}</option>`;
  }
};

const classifyQuery = async () => {
  const query = document.getElementById("filter-query").value.trim();
  const resultBox = document.getElementById("query-result");
  const refreshBtn = document.getElementById("btn-refresh");
  if (!query) {
    resultBox.textContent = "Nhập truy vấn để phân loại.";
    return;
  }

  resultBox.textContent = "Đang phân loại truy vấn...";
  try {
    const res = await fetch(`${API_BASE}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: query }),
    });
    if (!res.ok) throw new Error("Không thể phân loại");
    const data = await res.json();
    resultBox.innerHTML = `Truy vấn được xếp vào chủ đề <strong>${data.predicted_topic}</strong> với độ tin cậy ${(data.confidence * 100).toFixed(1)}%`;
  } catch (err) {
    resultBox.textContent = `Lỗi: ${err.message}`;
  }
};

const renderTopicsChart = (topics) => {
  const canvas = document.getElementById("stats-topics-chart");
  if (!canvas || !topics || !topics.length || typeof Chart === "undefined") return;

  const labels = topics.map((t) => t.topic);
  const dataCounts = topics.map((t) => t.count);

  const colors = [
    "#6366F1",
    "#EC4899",
    "#F97316",
    "#22C55E",
    "#EAB308",
    "#06B6D4",
    "#A855F7",
    "#F97373",
    "#10B981",
    "#3B82F6",
  ];

  if (topicsChart) {
    topicsChart.destroy();
  }

  topicsChart = new Chart(canvas, {
    type: "pie",
    data: {
      labels,
      datasets: [
        {
          data: dataCounts,
          backgroundColor: labels.map((_, idx) => colors[idx % colors.length]),
        },
      ],
    },
    options: {
      plugins: {
        legend: { position: "bottom" },
      },
    },
  });
};

const renderSourcesChart = (sources) => {
  const canvas = document.getElementById("stats-sources-chart");
  if (!canvas || !sources || !sources.length || typeof Chart === "undefined") return;

  const labels = sources.map((s) => formatSource(s.source));
  const dataCounts = sources.map((s) => s.count);

  if (sourcesChart) {
    sourcesChart.destroy();
  }

  sourcesChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          data: dataCounts,
          backgroundColor: "#38BDF8",
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: { font: { size: 10 } },
        },
        y: {
          beginAtZero: true,
          ticks: { precision: 0 },
        },
      },
    },
  });
};

const loadStats = async () => {
  const hours = document.getElementById("stats-hours").value;
  const limit = document.getElementById("stats-limit").value;
  const topicList = document.getElementById("stats-topics");
  const sourceList = document.getElementById("stats-sources");
  const sourceTopicList = document.getElementById("stats-source-topics");
  const dailyTopicList = document.getElementById("stats-daily-topics");
  topicList.innerHTML = "Đang tải...";
  sourceList.innerHTML = "";
  if (sourceTopicList) {
    sourceTopicList.innerHTML = "";
  }
  if (dailyTopicList) {
    dailyTopicList.innerHTML = "";
  }

  const params = new URLSearchParams({ hours, limit });
  try {
    const res = await fetch(`${API_BASE}/stats?${params.toString()}`);
    if (!res.ok) throw new Error("Không lấy được thống kê");
    const data = await res.json();

    topicList.innerHTML = data.topics.length
      ? data.topics.map((t) => `<li>${t.topic} (${t.count})</li>`).join("")
      : "<li>Chưa có dữ liệu</li>";
    sourceList.innerHTML = data.sources.length
      ? data.sources.map((s) => `<li>${formatSource(s.source)} (${s.count})</li>`).join("")
      : "<li>Chưa có dữ liệu</li>";

    if (data.topics && data.topics.length) {
      renderTopicsChart(data.topics);
    } else if (topicsChart) {
      topicsChart.destroy();
      topicsChart = null;
    }

    if (data.sources && data.sources.length) {
      renderSourcesChart(data.sources);
    } else if (sourcesChart) {
      sourcesChart.destroy();
      sourcesChart = null;
    }
    if (sourceTopicList) {
      sourceTopicList.innerHTML =
        data.source_topics && data.source_topics.length
          ? data.source_topics
              .map(
                (st) =>
                  `<li>${formatSource(st.source)}: <strong>${st.topic}</strong> (${st.count})</li>`
              )
              .join("")
          : "<li>Chưa có dữ liệu</li>";
    }

    // Load daily top topics (last 7 days by default)
    if (dailyTopicList) {
      dailyTopicList.innerHTML = "Đang tải...";
      try {
        const dailyRes = await fetch(`${API_BASE}/stats/daily-topics?days=7`);
        if (!dailyRes.ok) throw new Error("Không lấy được thống kê theo ngày");
        const dailyData = await dailyRes.json();
        dailyTopicList.innerHTML =
          dailyData.days && dailyData.days.length
            ? dailyData.days
                .map(
                  (d) =>
                    `<li>${d.date}: <strong>${d.topic}</strong> (${d.count})</li>`
                )
                .join("")
            : "<li>Chưa có dữ liệu</li>";
      } catch (err) {
        dailyTopicList.innerHTML = `Lỗi: ${err.message}`;
      }
    }
  } catch (err) {
    topicList.innerHTML = `Lỗi: ${err.message}`;
    sourceList.innerHTML = "";
    if (sourceTopicList) {
      sourceTopicList.innerHTML = "";
    }
    if (dailyTopicList) {
      dailyTopicList.innerHTML = "";
    }
  }
};

const createArticle = async () => {
  const title = document.getElementById("article-title").value.trim();
  const source = document.getElementById("article-source").value.trim();
  const url = document.getElementById("article-url").value.trim();
  const content = document.getElementById("article-content").value.trim();
  const classify = document.getElementById("article-classify").checked;
  const resultBox = document.getElementById("create-result");

  if (!title || !source || !url || !content) {
    resultBox.textContent = "Vui lòng nhập đầy đủ thông tin.";
    return;
  }

  resultBox.textContent = "Đang lưu...";
  try {
    const res = await fetch(`${API_BASE}/news`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        source,
        url,
        content,
        classify,
      }),
    });

    if (!res.ok) throw new Error("Không thể lưu bài viết");

    const data = await res.json();
    resultBox.innerHTML = `Đã lưu bài #${data.id} - Chủ đề: ${data.topic || "chưa xác định"}`;
    await loadNews();
  } catch (err) {
    resultBox.textContent = `Lỗi: ${err.message}`;
  }
};

const renderNewsItem = (article) => {
  const container = document.createElement("div");
  container.className = "news-item";

  const snippet = cleanHtml(article.summary || article.content || "");

  container.innerHTML = `
    <h3>${article.title}</h3>
    <div class="meta">Nguồn: ${formatSource(article.source)} · <a href="${article.url}" target="_blank">Link</a></div>
    ${article.topic ? `<span class="topic-tag">${article.topic}</span>` : ""}
    <p>${snippet.slice(0, 180)}...</p>
  `;

  return container;
};

const loadNews = async () => {
  const topic = document.getElementById("filter-topic").value.trim();
  const source = document.getElementById("filter-source").value.trim();
  const query = document.getElementById("filter-query").value.trim();
  const list = document.getElementById("news-list");
  const pageInfo = document.getElementById("news-page-info");
  const prevBtn = document.getElementById("news-prev");
  const nextBtn = document.getElementById("news-next");

  list.innerHTML = "Đang tải...";

  try {
    const params = new URLSearchParams();
    if (topic) params.append("topic", topic);
    if (source) params.append("source", source);
    if (query) params.append("query", query);
    const offset = (newsPage - 1) * NEWS_PAGE_SIZE;
    params.append("limit", String(NEWS_PAGE_SIZE));
    params.append("offset", String(offset));
    const res = await fetch(`${API_BASE}/news?${params.toString()}`);
    if (!res.ok) throw new Error("Không tải được danh sách tin");

    const data = await res.json();
    list.innerHTML = "";
    if (!data.items.length) {
      list.textContent = "Chưa có dữ liệu.";
      return;
    }

    data.items.forEach((article) => list.appendChild(renderNewsItem(article)));

    newsTotal = data.total ?? data.items.length;
    const totalPages = Math.max(1, Math.ceil(newsTotal / NEWS_PAGE_SIZE));
    if (pageInfo) {
      pageInfo.textContent = `Trang ${newsPage}/${totalPages}`;
    }
    if (prevBtn) {
      prevBtn.disabled = newsPage <= 1;
    }
    if (nextBtn) {
      nextBtn.disabled = newsPage >= totalPages;
    }
  } catch (err) {
    list.textContent = `Lỗi: ${err.message}`;
  }
};

const init = () => {
  document.getElementById("btn-classify").addEventListener("click", classifyText);
  document.getElementById("btn-create").addEventListener("click", createArticle);
  const refreshBtn = document.getElementById("btn-refresh");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      newsPage = 1;
      loadNews();
    });
  }
  document.getElementById("btn-classify-query").addEventListener("click", classifyQuery);
  document.getElementById("btn-stats").addEventListener("click", loadStats);
  document.getElementById("btn-topic-refresh").addEventListener("click", loadTopicArticles);
  document.getElementById("topic-select").addEventListener("change", loadTopicArticles);

  loadNews();
  loadStats();
  loadTopics().then(() => loadTopicArticles());
};

window.addEventListener("DOMContentLoaded", init);
