import datetime
import json
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from http import HTTPStatus
from typing import Any, Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

from htmls import date, footer_html, header_html


def get_sources_from_txt():
    sources = json.load(open("assets/news_sources.json", "r"))
    return ",".join([i.strip() for i in sources.keys()])


def get_from_worldnewsapi_com():
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
        params.update({"offset": offset, "number": news_items_per_call})
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
    df.to_json(
        f"assets/news.json",
        index=False,
        orient="records",
    )


def rank_news():
    news_items = [
        i
        for i in json.load(open("assets/news.json", "r"))
        if i.get("title") is not None and i.get("summary") is not None
    ]

    # TF IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words="english")
    titles = [i.get("title") for i in news_items]
    summaries = [i.get("summary").lower() for i in news_items]

    # Fit TF-IDF Model
    title_tfidf_matrix = tfidf.fit_transform(titles)
    summary_tfidf_matrix = tfidf.fit_transform(summaries)

    # Normalize and Extract Scores
    title_scores = np.mean(title_tfidf_matrix.toarray(), axis=1)
    summary_scores = np.mean(summary_tfidf_matrix.toarray(), axis=1)
    sentiment_scores = [i.get("sentiment") for i in news_items]
    source_scores = json.load(open("assets/news_sources.json", "r"))

    # Compute final relevance
    weights = {"title": 0.3, "source": 0.3, "summary": 0.2, "sentiment": 0.2}
    relevance_scores = []

    for i, article in enumerate(news_items):
        source_score = source_scores.get(
            "https://" + article["url"].replace("https://", "").split("/")[0]
        )
        score = (
            weights["title"] * title_scores[i] * 100
            + weights["source"] * source_score
            + weights["summary"] * summary_scores[i] * 100
            + weights["sentiment"] * sentiment_scores[i] * 100
        )
        relevance_scores.append(score)

    # Add scores to news items and sort
    for i, article in enumerate(news_items):
        article["relevance_score"] = relevance_scores[i]

    sorted_articles = sorted(
        news_items, key=lambda x: x["relevance_score"], reverse=True
    )
    return sorted_articles


def render_news(article_list: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    rank = 1
    max_rank = 15 if datetime.date.today().weekday() <= 4 else 30
    article_divs = [f"<p>Today's top {max_rank} stories</p>"]
    rendered_articles = []
    for article in article_list[:max_rank]:
        div = f"""
            <a href={article.get("url")} class="article-card" target="_blank">
                <div class="article-number">{rank}</div>
                <div class="article-content">
                    <div class="article-title">{article.get("title")}</div>
                    <div class="article-summary">{article.get("summary")}</div>
                </div>
            </a>
            """
        article_divs.append(div)
        rendered_articles.append(article.get("id"))
        rank += 1

    article_html = header_html + "\n".join(article_divs) + footer_html
    with open("src/sample.html", "w") as f:
        f.write(article_html)
    remaining = [i for i in article_list if i.get("id") not in rendered_articles]
    return article_html, remaining


def send_email(content: str):
    sender_email = os.environ.get("SENDER_EMAIL")
    app_password = os.environ.get("GMAIL_APP_PASSWORD_MAIN")
    recipients = json.loads(os.environ.get("RECIPIENTS"))

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


if __name__ == "__main__":
    load_dotenv()
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("vader_lexicon")
    print("Running script")
    # get_from_worldnewsapi_com()
    sorted_news = rank_news()
    complete_html, skipped_articles = render_news(article_list=sorted_news)
    send_email(content=complete_html)
    print("Emails sent :)")
