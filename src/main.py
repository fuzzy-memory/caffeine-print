import datetime
import json
import os

from dotenv import load_dotenv
import requests
from http import HTTPStatus


def get_sources_from_txt():
    with open("assets/news_sources.txt", "r") as f:
        sources = f.readlines()
    return ",".join([i.strip() for i in sources])


def get_from_worldnewsapi_com():
    # todo debug status vars
    # Set vars
    api_key = os.environ.get("WORLDNEWSAPI_KEY_CLIP")
    url = "https://api.worldnewsapi.com/search-news"
    # Date limit vars
    today = datetime.date.today()
    latest = datetime.datetime.combine(
        date=today - datetime.timedelta(days=0), time=datetime.time(0, 0, 0)
    )
    earliest = datetime.datetime.combine(
        date=today - datetime.timedelta(days=1), time=datetime.time(0, 0, 0)
    )
    offset = 0
    news_items_per_call = 100
    news_sources = get_sources_from_txt()

    # API ops
    print("Sending GET requests")
    headers = {"x-api-key": api_key}
    params = {
        "source-country": "in",
        "language": "en",
        "earliest-publish-date": earliest,
        "latest-publish-date": latest,
        "categories": "politics",
        "number": news_items_per_call,
        "offset": offset,
        "sort": "publish-time",
        "sort-direction": "DESC",
        "news-sources": news_sources,
    }

    # First call
    first_response = requests.get(url, headers=headers, params=params)
    if first_response.status_code == HTTPStatus.OK:
        news_items = first_response.json().get("news")
        offset += news_items_per_call
        total_items = first_response.json().get("available")
        remaining_items = total_items - offset
        remaining_quota = float(first_response.headers.get("X-API-Quota-Left"))
        print(
            f"Found {first_response.json().get('available')} articles\nRetrieved {offset} items in total. {remaining_items} items to go"
        )
    else:
        raise requests.HTTPError(
            f"{first_response.status_code}: {first_response.json().get('message')}"
        )

    # Pagination calls
    while True:
        if remaining_items <= 0 or remaining_quota <= 1:
            break
        next_response = requests.get(url, headers=headers, params=params)
        if next_response.status_code == HTTPStatus.OK:
            received = len(next_response.json().get("news"))
            if total_items - received < news_items_per_call:
                news_items_per_call = total_items - received
            news_items.extend(next_response.json().get("news"))
            remaining_items -= received
            offset += news_items_per_call
            remaining_quota = float(first_response.headers.get("X-API-Quota-Left"))
            print(
                f"Retrieved {len(news_items)} items in total."
                + (f"{remaining_items} to go" if remaining_items >= 0 else "")
            )
        else:
            json.dump(news_items, open("assets/incomplete_news.json", "w"))
            raise requests.HTTPError(
                f"{next_response.status_code}: Retrieved until offset {offset}. Quota: {next_response.headers.get('X-API-Quota-Left')}"
            )
    print(f"Operation complete. Remaining quota: {remaining_quota}")
    json.dump(news_items, open(f"assets/news.json", "w"))


def rank_news():
    import pandas as pd

    news_json = json.load(open("assets/news.json", "r"))
    df = pd.read_json("assets/news.json")
    df["source"] = df.apply(
        lambda x: x["url"].replace("https://", "").replace("www.", "").split("/")[0],
        axis=1,
    )
    print(
        df[
            ["id", "title", "publish_date", "sentiment", "source", "summary"]
        ].to_string()
    )


if __name__ == "__main__":
    load_dotenv()
    print("Running script")
    # get_from_worldnewsapi_com()
    rank_news()
