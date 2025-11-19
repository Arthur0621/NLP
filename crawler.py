"""
Script crawler để lấy nội dung bài báo từ các trang báo điện tử Việt Nam
"""
import requests
from bs4 import BeautifulSoup
import json
import time
from tqdm import tqdm
import re

class NewsCrawler:
    """Class để crawl tin tức từ các trang báo"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def crawl_vnexpress(self, url):
        """Crawl bài báo từ VnExpress"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Lấy tiêu đề
            title_tag = soup.find('h1', class_='title-detail')
            title = title_tag.get_text(strip=True) if title_tag else ""
            
            # Lấy mô tả
            description_tag = soup.find('p', class_='description')
            description = description_tag.get_text(strip=True) if description_tag else ""
            
            # Lấy nội dung
            content_div = soup.find('article', class_='fck_detail')
            if content_div:
                paragraphs = content_div.find_all('p', class_='Normal')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                content = ""
            
            full_content = f"{description} {content}".strip()
            
            return {
                'title': title,
                'content': full_content,
                'url': url,
                'source': 'vnexpress'
            }
        except Exception as e:
            print(f"Lỗi khi crawl VnExpress {url}: {str(e)}")
            return None
    
    def crawl_tuoitre(self, url):
        """Crawl bài báo từ Tuổi Trẻ"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Lấy tiêu đề
            title_tag = soup.find('h1', class_='detail-title')
            title = title_tag.get_text(strip=True) if title_tag else ""
            
            # Lấy mô tả
            description_tag = soup.find('h2', class_='detail-sapo')
            description = description_tag.get_text(strip=True) if description_tag else ""
            
            # Lấy nội dung
            content_div = soup.find('div', id='main-detail-body')
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                content = ""
            
            full_content = f"{description} {content}".strip()
            
            return {
                'title': title,
                'content': full_content,
                'url': url,
                'source': 'tuoitre'
            }
        except Exception as e:
            print(f"Lỗi khi crawl Tuổi Trẻ {url}: {str(e)}")
            return None
    
    def crawl_thanhnien(self, url):
        """Crawl bài báo từ Thanh Niên"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Lấy tiêu đề
            title_tag = soup.find('h1', class_='details__headline')
            title = title_tag.get_text(strip=True) if title_tag else ""
            
            # Lấy mô tả
            description_tag = soup.find('div', class_='details__summary')
            description = description_tag.get_text(strip=True) if description_tag else ""
            
            # Lấy nội dung
            content_div = soup.find('div', id='contentDetail')
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                content = ""
            
            full_content = f"{description} {content}".strip()
            
            return {
                'title': title,
                'content': full_content,
                'url': url,
                'source': 'thanhnien'
            }
        except Exception as e:
            print(f"Lỗi khi crawl Thanh Niên {url}: {str(e)}")
            return None
    
    def crawl_dantri(self, url):
        """Crawl bài báo từ Dân Trí"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Lấy tiêu đề
            title_tag = soup.find('h1', class_='dt-news__title')
            title = title_tag.get_text(strip=True) if title_tag else ""
            
            # Lấy mô tả
            description_tag = soup.find('h2', class_='dt-news__sapo')
            description = description_tag.get_text(strip=True) if description_tag else ""
            
            # Lấy nội dung
            content_div = soup.find('div', class_='dt-news__content')
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                content = ""
            
            full_content = f"{description} {content}".strip()
            
            return {
                'title': title,
                'content': full_content,
                'url': url,
                'source': 'dantri'
            }
        except Exception as e:
            print(f"Lỗi khi crawl Dân Trí {url}: {str(e)}")
            return None
    
    def crawl_generic(self, url):
        """Crawl bài báo từ trang web bất kỳ (phương pháp chung)"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Lấy tiêu đề (thử nhiều cách)
            title = ""
            title_tags = [
                soup.find('h1'),
                soup.find('meta', property='og:title'),
                soup.find('title')
            ]
            for tag in title_tags:
                if tag:
                    title = tag.get('content', '') if tag.name == 'meta' else tag.get_text(strip=True)
                    if title:
                        break
            
            # Lấy nội dung (thử nhiều cách)
            content = ""
            
            # Thử tìm article tag
            article = soup.find('article')
            if article:
                paragraphs = article.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Nếu không có, thử tìm div có class chứa 'content' hoặc 'article'
            if not content:
                content_divs = soup.find_all('div', class_=re.compile(r'(content|article|body|detail)', re.I))
                for div in content_divs:
                    paragraphs = div.find_all('p')
                    if len(paragraphs) > 3:  # Chỉ lấy div có nhiều đoạn văn
                        content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                        break
            
            # Nếu vẫn không có, lấy tất cả thẻ p
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs[:20]])  # Giới hạn 20 đoạn đầu
            
            return {
                'title': title,
                'content': content,
                'url': url,
                'source': 'generic'
            }
        except Exception as e:
            print(f"Lỗi khi crawl {url}: {str(e)}")
            return None
    
    def crawl_url(self, url):
        """
        Crawl bài báo từ URL, tự động nhận diện nguồn
        """
        # Xác định nguồn từ URL
        if 'vnexpress.net' in url:
            return self.crawl_vnexpress(url)
        elif 'tuoitre.vn' in url:
            return self.crawl_tuoitre(url)
        elif 'thanhnien.vn' in url:
            return self.crawl_thanhnien(url)
        elif 'dantri.com.vn' in url:
            return self.crawl_dantri(url)
        else:
            # Sử dụng phương pháp chung cho các trang khác
            return self.crawl_generic(url)
    
    def crawl_multiple_urls(self, urls, delay=1):
        """
        Crawl nhiều URL
        
        Args:
            urls: List các URL cần crawl
            delay: Thời gian chờ giữa các request (giây)
        
        Returns:
            list: Danh sách kết quả crawl
        """
        results = []
        
        for url in tqdm(urls, desc="Crawling"):
            result = self.crawl_url(url)
            if result:
                results.append(result)
            
            # Chờ để tránh bị chặn
            time.sleep(delay)
        
        return results

def crawl_from_dataset(input_file, output_file, max_urls=100):
    """
    Đọc URL từ dataset và crawl lại nội dung
    
    Args:
        input_file: File JSON chứa dataset gốc
        output_file: File JSON để lưu kết quả crawl
        max_urls: Số lượng URL tối đa cần crawl (để test)
    """
    print(f"Đang đọc URLs từ {input_file}...")
    
    # Đọc dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Lấy danh sách URLs
    urls = []
    for item in data[:max_urls]:
        if 'url' in item and item['url']:
            urls.append({
                'url': item['url'],
                'topic': item.get('topic', ''),
                'original_title': item.get('title', '')
            })
    
    print(f"Tổng số URLs cần crawl: {len(urls)}")
    
    # Khởi tạo crawler
    crawler = NewsCrawler()
    
    # Crawl
    results = []
    for item in tqdm(urls, desc="Crawling"):
        result = crawler.crawl_url(item['url'])
        if result and result['content']:
            result['topic'] = item['topic']
            results.append(result)
        
        # Chờ 1-2 giây giữa các request
        time.sleep(1)
    
    # Lưu kết quả
    print(f"\nĐã crawl thành công {len(results)}/{len(urls)} URLs")
    print(f"Đang lưu kết quả vào {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Hoàn thành!")

def main():
    """Hàm main để test crawler"""
    # Test với một số URL mẫu
    test_urls = [
        'https://vnexpress.net/nguoi-chet-trong-mua-lu-nghin-nam-co-mot-o-my-tang-len-28-4494262.html',
        'https://tuoitre.vn/duc-co-the-suy-thoai-kinh-te-nam-nay-20220801084837232.htm',
        'https://thanhnien.vn/dia-phuong-nao-dung-dau-ca-nuoc-tong-diem-3-mon-van-toan-ngoai-ngu-post1483653.html'
    ]
    
    crawler = NewsCrawler()
    
    print("="*70)
    print("DEMO CRAWLER TIN TỨC")
    print("="*70)
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n{i}. Đang crawl: {url}")
        result = crawler.crawl_url(url)
        
        if result:
            print(f"   ✓ Tiêu đề: {result['title'][:100]}...")
            print(f"   ✓ Độ dài nội dung: {len(result['content'])} ký tự")
            print(f"   ✓ Nguồn: {result['source']}")
        else:
            print(f"   ✗ Không crawl được")
        
        time.sleep(1)

if __name__ == "__main__":
    # Chạy demo
    main()
    
    # Hoặc crawl từ dataset (uncomment để sử dụng)
    # crawl_from_dataset('Dataset/news_dataset.json', 'Dataset/crawled_news.json', max_urls=100)
