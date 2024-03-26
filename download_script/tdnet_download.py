import os
import json
import time
import random
import logging
import requests
from pypdf import PdfReader
from tqdm import tqdm

# ログの設定
logging.basicConfig(filename="/root/src/log/download.log", level=logging.INFO)

class TdnetDownloader:
    def __init__(self, max_retries=5):
        self.max_retries = max_retries

    def fetch_data(self, url):
        """指定されたURLからデータを取得する"""
        response = requests.get(url)
        data = json.loads(response.content)
        return [item["Tdnet"] for item in data["items"]]

    def extract_info(self, data):
        """取得したデータから必要な情報を抽出する"""
        return [
            {
                "pubdate": item["pubdate"],
                "company_code": item["company_code"],
                "document_url": item["document_url"][len("https://webapi.yanoshin.jp/rd.php?"):]
            } for item in data]

    def save_data(self, data, file_path):
        """データをJSONファイルに保存する"""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def download_pdf(self, url):
        """指定されたURLからPDFファイルをダウンロードする"""
        response = requests.get(url)
        time.sleep(random.randint(6, 15))  # ダウンロードの間隔を設定
        return response.content

    def save_pdf(self, content, file_path):
        """ダウンロードしたPDFファイルを保存する"""
        with open(file_path, "wb") as f:
            f.write(content)

    def validate_pdf(self, file_path):
        """PDFファイルが正しいかを検証する"""
        try:
            with open(file_path, "rb") as f:
                pdf = PdfReader(f)
                return len(pdf.pages) > 0
        except Exception as e:
            logging.error(f"Error validating PDF at {file_path}: {e}")
            return False

    def process_downloads(self, data):
        """PDFファイルのダウンロードと保存を処理する"""
        total = len(data)
        progress_bar = tqdm(total=total, desc="Downloading PDFs", unit="file")
        failed_downloads = []

        for item in data:
            if item["company_code"][-1] == "0":
                retries = 0
                success = False

                while retries < self.max_retries and not success:
                    try:
                        pdf_content = self.download_pdf(item["document_url"])
                        date = item["pubdate"].split(" ")[0]
                        file_name = f"{date}_{item['company_code']}.pdf"
                        file_path = f"/root/src/raw_data/timely-disclosure/{file_name}"

                        self.save_pdf(pdf_content, file_path)

                        if self.validate_pdf(file_path):
                            success = True
                            progress_bar.update(1)
                            logging.info(f"Downloaded {file_path}.")
                        else:
                            retries += 1
                            os.remove(file_path)  # 壊れたファイルを削除
                            logging.warning(f"Failed to validate {file_path}. Retrying download.")

                    except Exception as e:
                        logging.error(f"Failed to download {item['document_url']}: {e}")
                        retries += 1

                if not success:
                    failed_downloads.append(item)

        progress_bar.close()
        return failed_downloads

    def retry_failed_downloads(self, failed_downloads):
        """ダウンロードに失敗したファイルを再度ダウンロードする"""
        if not failed_downloads:
            logging.info("No failed downloads to retry.")
            return

        self.save_data(failed_downloads, "/root/src/log/path_to_failed_downloads.json")
        self.process_downloads(failed_downloads)

    def run(self, mode="today"):
        """
        TDnetからデータを取得し、PDFファイルをダウンロードする
        mode: "today" - 今日の日付だけダウンロード, "month" - 過去一か月分をダウンロード
        """
        if mode == "today":
            start_date = end_date = time.strftime("%Y%m%d")
        elif mode == "month":
            end_date = time.strftime("%Y%m%d")
            start_date = time.strftime("%Y%m%d", time.localtime(time.time() - 60 * 60 * 24 * 30))
        else:
            raise ValueError("Invalid mode. Choose either 'today' or 'month'.")

        url = f"https://webapi.yanoshin.jp/webapi/tdnet/list/{start_date}-{end_date}.json?limit=10000"
        data = self.fetch_data(url)
        extracted_data = self.extract_info(data)
        self.save_data(extracted_data, "/root/src/log/path_to_extracted_data.json")
        failed_downloads = self.process_downloads(extracted_data)
        self.retry_failed_downloads(failed_downloads)

if __name__ == "__main__":
    downloader = TdnetDownloader(max_retries=5)
    downloader.run()