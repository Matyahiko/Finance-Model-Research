import time
import requests
import json
import random
import os
from pypdf import PdfReader
import schedule
from tqdm import tqdm
import logging

# ログの設定
logging.basicConfig(filename='download.log', level=logging.INFO)

class TdnetDownloader:
    def __init__(self, max_retries=5):
        # 最大リトライ回数をセット
        self.max_retries = max_retries

    def get_url(self, URL):
        # URLからデータを取得するメソッド
        res = requests.get(URL)
        response_loads = json.loads(res.content)
        data_list = [data["Tdnet"] for data in response_loads["items"]]

        # データから必要な情報を抽出
        extracted_data = [
            {
                "pubdate": item["pubdate"],
                "company_code": item["company_code"],
                "document_url": item["document_url"]
            } for item in data_list]

        # 抽出したデータを保存
        with open("path_to_extracted_data.json", "w") as f:
            json.dump(extracted_data, f, indent=4)

        return extracted_data
    
    def download_pdf(self, data):
        # PDFファイルをダウンロードするメソッド
        total = len(data)
        progress_bar = tqdm(total=total, desc='Downloading PDFs', unit='file')
        failed_downloads = []

        for d in data:
            if d["company_code"][-1] == "0":
                retries = 0
                success = False

                while retries < self.max_retries and not success:
                    try:
                        pdf = requests.get(d["document_url"])
                        # ダウンロードの間隔を設定
                        time.sleep(random.randint(6, 15))

                        date = d["pubdate"].split(" ")
                        name = date[0] + "_" + d["company_code"]
                        file_path = f"../nas/tdnet/{name}.pdf"

                        # ダウンロードした内容を保存
                        with open(file_path, "wb") as f:
                            f.write(pdf.content)

                        if self.validate_pdf(file_path):
                            success = True
                            progress_bar.update(1)
                            logging.info(f"Downloaded {file_path}.")
                        else:
                            retries += 1
                            os.remove(file_path)  # 壊れたファイルを削除
                            logging.warning(f"Failed to validate {file_path}. Retrying download.")

                    except Exception as e:
                        logging.error(f"Failed to download {d['document_url']}: {e}")
                        retries += 1

                if not success:
                    failed_downloads.append(d)

        progress_bar.close()

        # ダウンロードに失敗したデータを保存
        if failed_downloads:
            with open("path_to_failed_downloads.json", "w") as f:
                json.dump(failed_downloads, f, indent=4)

        return failed_downloads

    def validate_pdf(self, file_path):
        # PDFが正しいか確認するメソッド
        try:
            with open(file_path, "rb") as f:
                pdf = PdfReader(f)
                num_pages = len(pdf.pages)
            return num_pages > 0
        except Exception as e:
            print(f"Error validating PDF at {file_path}: {e}")
            return False

    def retry_failed_downloads(self):
        # ダウンロードに失敗したデータを再度ダウンロードするメソッド
        if not os.path.exists("path_to_failed_downloads.json"):
            print("No failed downloads to retry.")
            return

        with open("path_to_failed_downloads.json", "r") as f:
            failed_downloads = json.load(f)

        # 失敗したダウンロードの情報を削除
        os.remove("path_to_failed_downloads.json")

        # ダウンロードのリトライ
        self.download_pdf(failed_downloads)

    def run(self):
        # 実行するメソッド
        start_date = time.strftime("%Y%m%d", time.localtime(time.time() - 60 * 60 * 24 * 30))
        end_date = time.strftime("%Y%m%d", time.localtime(time.time()))

        limit = 10000

        URL = f"https://webapi.yanoshin.jp/webapi/tdnet/list/{start_date}-{end_date}.json?limit={limit}"

        # データの取得
        data = self.get_url(URL)

        # PDFのダウンロード
        self.download_pdf(data)

        # 失敗したダウンロードの再実行
        self.retry_failed_downloads()

if __name__ == "__main__":
    downloader = TdnetDownloader(max_retries=5)
    downloader.run()