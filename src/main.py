import datetime
import json
import os
import smtplib
import ssl
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import nltk  # type: ignore
from dotenv import load_dotenv

from db import retrieve_recipients
from get_news import get_news_from_api
from htmls import get_formatted_date, render_news
from ranking import rank_articles
from similarity_clustering import deduplicate_articles


def send_email(content: str, test_mode: bool):
    date = get_formatted_date()

    sender_email = os.environ.get("SENDER_EMAIL")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    recipients = retrieve_recipients(test_mode=test_mode)
    if not sender_email or not app_password or not recipients:
        raise ValueError("All required mailing vars not found")

    # Set up SMTP object
    smtp_server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    smtp_server.ehlo()
    smtp_server.login(sender_email, app_password)

    # Set up message
    message = MIMEMultipart("alternative")
    message["Subject"] = f"Hashbrown: {date}"
    message["From"] = sender_email
    message.attach(MIMEText(content, "html"))

    # Send email
    for recipient in recipients:
        smtp_server.sendmail(
            from_addr=sender_email, to_addrs=recipient, msg=message.as_string()
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
    get_news_from_api(test_mode=testing_flag)
    print("----------")
    dedup_processed_news = deduplicate_articles(test_mode=testing_flag)
    print("----------")
    sorted_news = rank_articles(news_items=dedup_processed_news, test_mode=testing_flag)
    print("----------")
    complete_html, skipped_articles = render_news(
        article_df=sorted_news,
        test_mode=testing_flag,
    )
    send_email(content=complete_html, test_mode=testing_flag)
