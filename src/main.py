import datetime
import json
import os
import smtplib
import ssl
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from http import HTTPStatus
from typing import Any, Dict

import nltk  # type: ignore
import pandas as pd
import requests
from dotenv import load_dotenv

from htmls import get_formatted_date, render_news
from ranking import rank_articles
from similarity_clustering import deduplicate_articles


def get_from_worldnewsapi_com(test_mode: bool = False):
    if test_mode:
        dir_path = "assets/test/"
        pth = os.path.join(os.path.curdir, dir_path)
        if "news.json" in os.listdir(pth):
            item_count = len(json.load(open("assets/test/news.json", "r")))
            print(f"{item_count} articles already extracted in `assets/test`")
            return

    # Set vars
    api_key = os.environ.get("WORLDNEWSAPI_KEY")
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
    news_sources = ",".join(
        [i.strip() for i in json.load(open("assets/news_sources.json", "r")).keys()]
    )

    # API ops
    print("Sending GET requests")
    headers = {"x-api-key": api_key}
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

    # First call
    first_response = requests.get(url, headers=headers, params=params)
    if first_response.status_code == HTTPStatus.OK:
        news_items = first_response.json().get("news")
        offset += news_items_per_call
        total_items = first_response.json().get("available")
        remaining_items = total_items - offset
        remaining_quota = float(first_response.headers.get("X-API-Quota-Left"))  # type: ignore
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
    print(f"Operation complete. Remaining quota: {remaining_quota}")
    df = pd.DataFrame(news_items).drop_duplicates(subset=["id"])
    path_to_write = "assets/" + ("test/" if test_mode else "") + "news.json"
    df.to_json(
        path_to_write,
        index=False,
        orient="records",
    )


def send_email(content: str):
    date = get_formatted_date()

    sender_email = os.environ.get("SENDER_EMAIL")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    recipients_raw = os.environ.get("RECIPIENTS")
    if not sender_email or not app_password or not recipients_raw:
        raise ValueError("All required mailing vars not found in .env")
    recipients = json.loads(recipients_raw)

    # Set up SMTP object
    smtp_server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    smtp_server.ehlo()
    smtp_server.login(sender_email, app_password)

    # Set up message
    message = MIMEMultipart("alternative")
    message["Subject"] = f"Hashbrown: {date}"
    message["From"] = sender_email
    message["To"] = ", ".join(recipients)
    message.attach(MIMEText(content, "html"))

    # Send email
    smtp_server.sendmail(
        from_addr=sender_email, to_addrs=recipients, msg=message.as_string()
    )
    smtp_server.close()
    print("Emails sent :)")


if __name__ == "__main__":
    testing_flag = sys.argv[1] == "1"  # Cmd line arg; Set as 1 to enable testing mode
    load_dotenv()
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("stopwords", quiet=True)

    if testing_flag:
        print("----- RUNNING IN TEST MODE -----")
    else:
        print("----- Running script -----")
    get_from_worldnewsapi_com(test_mode=testing_flag)
    print("----------")
    dedup_processed_news = deduplicate_articles(test_mode=testing_flag)
    print("----------")
    sorted_news = rank_articles(news_items=dedup_processed_news, test_mode=testing_flag)
    print("----------")
    complete_html, skipped_articles = render_news(
        article_df=sorted_news,
        test_mode=testing_flag,
    )
    # send_email(content=complete_html)
