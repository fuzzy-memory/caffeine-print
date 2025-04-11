import datetime
import json
import os
from http import HTTPStatus
from typing import Any, Dict, Literal, Union

import pandas as pd
import requests
from dateutil import relativedelta

from properties import literature_and_laurels, permitted_callers


def generate_general_news_request_params(
    key: str,
) -> Dict[str, Union[Dict[str, str], int]]:
    # Date limit vars
    today = datetime.date.today()
    latest = datetime.datetime.combine(
        date=today - datetime.timedelta(days=1), time=datetime.time(23, 59, 59)
    ).strftime("%Y-%m-%d %H:%M:%S")
    earliest = datetime.datetime.combine(
        date=today - datetime.timedelta(days=1), time=datetime.time(0, 0, 0)
    ).strftime("%Y-%m-%d %H:%M:%S")
    offset = 0
    news_items_per_call = 100
    news_sources = ",".join(
        [i.strip() for i in json.load(open("assets/control.json", "r")).get("source_scores").keys()]
    )

    # API ops
    headers = {"x-api-key": key}
    params: Dict[str, Any] = {
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
    return {
        "headers": headers,
        "params": params,
        "offset": offset,
        "news_items_per_call": news_items_per_call,
    }


def generate_laurels_request_params(key) -> Dict[str, Union[Dict[str, str], int]]:
    # Date limit vars
    today = datetime.date.today()
    last_sunday = today + relativedelta.relativedelta(weekday=relativedelta.SU(-1))

    if last_sunday == today:
        last_sunday -= datetime.timedelta(days=7)

    latest = datetime.datetime.combine(
        date=today - datetime.timedelta(days=0), time=datetime.time(0, 0, 0)
    )
    earliest = datetime.datetime.combine(date=last_sunday, time=datetime.time(0, 0, 0))
    offset = 0
    news_items_per_call = 100

    # API ops
    headers = {"x-api-key": key}
    params: Dict[str, Any] = {
        "language": "en",
        "earliest-publish-date": earliest,
        "latest-publish-date": latest,
        "text": "winners of english literary awards OR pulitzer OR booker -oscars -grammy -emmy -movies -music",
        "number": news_items_per_call,
        "offset": offset,
        "sort": "publish-time",
        "sort-direction": "DESC",
        # "news-sources": news_sources,
    }
    return {
        "headers": headers,
        "params": params,
        "offset": offset,
        "news_items_per_call": news_items_per_call,
    }


def make_api_calls(
    *,
    headers: Dict[str, str],
    params: Dict[str, Any],
    offset: int,
    news_items_per_call: int,
    test_mode: bool,
    caller: str,
):
    assert caller in permitted_callers
    url = "https://api.worldnewsapi.com/search-news"

    # First call
    first_response = requests.get(url, headers=headers, params=params)
    if first_response.status_code == HTTPStatus.OK:
        news_items = first_response.json().get("news")
        total_items = first_response.json().get("available")
        remaining_quota = float(first_response.headers.get("X-API-Quota-Left"))  # type: ignore
        if total_items == 0:
            return [], remaining_quota
        offset += news_items_per_call
        remaining_items = total_items - offset
        if total_items < news_items_per_call:
            offset = total_items
            remaining_items = 0
        print(
            f"Found {first_response.json().get('available')} articles\nRetrieved {offset} items in total. {remaining_items} items to go"
        )
    else:
        raise requests.HTTPError(
            f"{first_response.status_code}: {first_response.json().get('message')}"
        )

    # Pagination calls
    while True:
        if test_mode:
            break
        if remaining_items <= 0 or remaining_quota <= 1:
            break
        params.update({"offset": offset, "number": news_items_per_call})
        next_response = requests.get(url, headers=headers, params=params)
        if next_response.status_code == HTTPStatus.OK:
            received = len(next_response.json().get("news"))
            if total_items - received < news_items_per_call:
                news_items_per_call = total_items - received
            news_items.extend(next_response.json().get("news"))
            remaining_items -= received
            offset += news_items_per_call
            remaining_quota = float(first_response.headers.get("X-API-Quota-Left"))  # type: ignore
            print(
                f"Retrieved {len(news_items)} items in total. "
                + (f"{remaining_items} to go" if remaining_items >= 0 else "")
            )
        else:
            json.dump(news_items, open("assets/incomplete_news.json", "w"))
            raise requests.HTTPError(
                f"{next_response.status_code}: Retrieved until offset {offset}. Quota: {next_response.headers.get('X-API-Quota-Left')}"
            )
    for obj in news_items:
        obj.update({"api_query_category": caller})
    return news_items, remaining_quota


def get_news_from_api(test_mode: bool = False):
    if test_mode:
        dir_path = "assets/test/"
        pth = os.path.join(os.path.curdir, dir_path)
        if "news.json" in os.listdir(pth):
            item_count = len(json.load(open("assets/test/news.json", "r")))
            print(f"{item_count} articles already extracted in `assets/test`")
            return

    # Set vars
    api_key = os.environ.get("WORLDNEWSAPI_KEY")
    callers = dict(
        zip(
            permitted_callers,
            [generate_general_news_request_params, generate_laurels_request_params],
        )
    )
    news_items = []
    remaining_quota = 0.0
    if literature_and_laurels == 0:
        callers.pop("laurels")

    for caller in callers.keys():
        generated_request_body = callers.get(caller)(key=api_key)
        headers = generated_request_body.get("headers")
        params = generated_request_body.get("params")
        offset = generated_request_body.get("offset")
        news_items_per_call = generated_request_body.get("news_items_per_call")
        print(f"Sending GET requests for {caller} news items")
        api_output, remaining_quota = make_api_calls(
            headers=headers,
            params=params,
            offset=offset,
            news_items_per_call=news_items_per_call,
            test_mode=test_mode,
            caller=caller,
        )
        news_items.extend(api_output)

    print(f"Operation complete. Remaining quota: {remaining_quota}")
    if len(news_items) == 0:
        print("No news items returned from API")
        exit(1)
    df = pd.DataFrame(news_items).drop_duplicates(subset=["id"])
    path_to_write = "assets/" + ("test/" if test_mode else "") + "news.json"
    df.to_json(
        path_to_write,
        index=False,
        orient="records",
    )
