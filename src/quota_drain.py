import json
from http import HTTPStatus

from dotenv import load_dotenv
import os
import datetime

import requests

if __name__=="__main__":
    load_dotenv()
    print("Running script")
    api_key = os.environ.get("WORLDNEWSAPI_KEY_TECHNIQUEC")
    url = "https://api.worldnewsapi.com/search-news"
    today = datetime.date.today()
    yesterday = datetime.datetime.combine(date=today - datetime.timedelta(days=1), time=datetime.time(0, 0, 0))
    day_before = datetime.datetime.combine(date=today - datetime.timedelta(days=2), time=datetime.time(0, 0, 0))
    offset=0
    num=100
    print("Sending GET request")
    first_res=requests.get(url, headers={'x-api-key': api_key},
                 params={"source-country": "in", "language": "en", "earliest-publish-date": day_before,
                         "latest-publish-date": yesterday, "categories": "politics", "number": num, "offset": offset,
                         "sort-direction": "DESC"})
    if first_res.status_code==HTTPStatus.OK:
        print("First 100 news items received")
        res=first_res.json().get("news")
        offset += num

    else:
        print(first_res.json())
    while True:
        next_res=requests.get(url, headers={'x-api-key': api_key},
                 params={"source-country": "in", "language": "en", "earliest-publish-date": day_before,
                         "latest-publish-date": yesterday, "categories": "politics", "number": num, "offset": offset,
                         "sort-direction": "DESC"})
        if next_res.status_code==HTTPStatus.OK:
            res.extend(next_res.json().get("news"))
            offset+=num
        if float(next_res.headers.get('X-API-Quota-Left'))<=0.0:
            break
        print(f"Received {offset} items so far... Quota left: {next_res.headers.get('X-API-Quota-Left')}")

    json.dump(res, open("mad_long_news.json", "w"))