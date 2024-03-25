from newspaper import Article
import feedparser
import random
import time
import os
import csv
import json
import traceback

def debug_log(message):
    print(f"[DEBUG] {message}")

# 不正なファイル名文字を削除/置換する関数
def clean_title(title):
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        title = title.replace(char, '_')
    return title

def read_rss_list_from_json(json_file):
    with open(json_file, mode="r", encoding="utf-8") as file:
        rss_list = json.load(file)
    return rss_list

def random_sleep(min_sec=5, max_sec=15):
    sleep_time = random.uniform(min_sec, max_sec)
    time.sleep(sleep_time)

def fetch_article_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    article = Article(url)
    article.headers = headers
    article.download()
    article.parse()
    return article.text

def save_article_to_tsv(file_path, title, date, link, text):
    with open(file_path, "a", newline="", encoding="utf-8") as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerow([title, date, link, text])

if __name__ == "__main__":
    rss_list = read_rss_list_from_json("/root/src/download_script/RSS.json")
    debug_log(f"RSS List loaded: {rss_list}")

    for rss_item in rss_list:
        source_name = rss_item["name"]
        feed_url = rss_item["url"]
        debug_log(f"Processing RSS feed from {source_name}")
        
        # ソースごとのディレクトリを作成
        source_dir = os.path.join("raw_data/news", source_name)
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)

        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            debug_log(f"Failed to parse RSS from {feed_url}. Error: {e}")
            continue

        for index, entry in enumerate(feed.entries):
            title = entry.title
            cleaned_title = clean_title(title)  # タイトルをクリーンアップ
            
            if "published" in entry:
                date = entry.published.replace(':', '-')
            elif "updated" in entry:
                date = entry.updated.replace(':', '-')
            else:
                date = ""
            
            link = entry.link
            
            # 新しいTSVファイルパスを設定
            tsv_file_path = os.path.join(source_dir, f"{cleaned_title}.tsv")
            
            # ヘッダーを追加
            if not os.path.exists(tsv_file_path):
                with open(tsv_file_path, "w", newline="", encoding="utf-8") as tsv_file:
                    writer = csv.writer(tsv_file, delimiter='\t')
                    writer.writerow(["Title", "Date", "Link", "Text"])

            debug_log(f"Processing article {index + 1}/{len(feed.entries)}")
            debug_log(f"Title: {title}")
            debug_log(f"Date: {date}")
            debug_log(f"Link: {link}")

            try:
                text = fetch_article_text(link)
                debug_log(f"First 100 chars of the text: {text[:100]}")
                save_article_to_tsv(tsv_file_path, title, date, link, text)
                debug_log(f"Saved article to {tsv_file_path}")
            except Exception as e:
                debug_log(f"Failed to download article from {link}. Error: {e}")
                debug_log(traceback.format_exc())  # Print full error traceback
            
            random_sleep()