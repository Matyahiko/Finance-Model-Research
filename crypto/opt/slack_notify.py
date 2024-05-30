import requests

# Slackにmsgを通知するためのクラス
slack_notify_TOKEN = ""
SLACK_NOTIFY_TOKEN = ""


class SlackNotify:
    def __init__(self):
        self.slack_notify_token = SLACK_NOTIFY_TOKEN
        self.slack_notify_api = "slack_api"
        self.headers = {
            "Authorization": f"Bearer {self.slack_notify_token}"
        }
        
    def send(self, msg) -> str:
        msg = {"message": f"{msg}"}
        print(msg)
        requests.post(self.slack_notify_api, headers = self.headers, data = msg)