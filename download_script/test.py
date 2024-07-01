id = "azumaru_cc"
pw = "xR5GbgrS8"

import urllib3
import requests
http = urllib3.PoolManager()

url = "https://csvex.com/kabu.plus/csv/japan-all-stock-prices/daily/japan-all-stock-prices.csv"
headers = urllib3.util.make_headers(basic_auth="%s:%s" % (id, pw) )
response = requests.post(url, headers=headers)

print(response.text)

#with open("japan-all-stock-prices.csv", "wb") as f:
#    f.write(str(response.text))
#f = open("japan-all-stock-prices.csv", "wb")
#f.write(response.text)
#f.close()

url = "https://csvex.com/kabu.plus/json/japan-all-stock-prices/daily/japan-all-stock-prices.json"
headers = urllib3.util.make_headers(basic_auth="%s:%s" % (id, pw) )
response = requests.post(url, headers=headers)

with open("japan-all-stock-prices.csv", "wb") as f:
    f.write(response.text)

#f = open("japan-all-stock-prices.json", "wb")
#f.write(response.text)
#f.close()