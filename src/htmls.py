import datetime
import json
import re
from typing import List, Tuple

import pandas as pd

from data_models import Article


def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix


def get_formatted_date() -> str:
    today = datetime.date.today()
    prefix = today.strftime(f"%A, %B")
    suffix = datetime.date.today().strftime("%Y")
    final_date = f"{prefix} {ordinal(today.day)} {suffix}"
    return final_date


def make_base_html():
    date = get_formatted_date()
    header_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hash Brown: {date}</title>
        <style>
            body {{
                background-color: #121212;
                color: white;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 800px;
                margin: auto;
            }}
            .article-card {{
                display: flex;
                align-items: center;
                background-color: #1e1e1e;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(255, 255, 255, 0.1);
                text-decoration: none;
                color: white;
                border: none;
                position: relative;
            }}
            .article-card:hover {{
                background-color: #292929;
            }}
            .article-card:visited, .article-card:link, .article-card:active {{
                color: white;
                text-decoration: none;
            }}
            .article-number {{
                display: flex;
                justify-content: center;
                align-items: center;
                width: 50px;
                font-size: 24px;
                font-weight: bold;
                color: #bbbbbb;
                background-color: #292929;
                height: 100%;
                border-top-left-radius: 8px;
                border-bottom-left-radius: 8px;
            }}
            .article-content {{
                flex: 1;
                padding-left: 15px;
            }}
            .article-title {{
                font-size: 20px;
                font-weight: bold;
            }}
            .article-summary {{
                color: #bbbbbb;
                margin-top: 5px;
            }}
            .article-image {{
                width: 120px;
                height: 80px;
                object-fit: cover;
                max-width: 120px;
                max-height: 80px;
                border-radius: 8px;
                margin-left: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Hash Brown: {date}</h1>
    """

    footer_html = """
        <center>Made with ❤️ in Mumbai</center>
        </div>
    </body>
    </html>
    """
    return header_html, footer_html


def render_news(
    article_df: pd.DataFrame,
    test_mode: bool = False,
) -> Tuple[str, List[Article]]:
    article_list = [
        Article(**json.loads(i.to_json())) for _, i in article_df.iterrows()
    ]
    rank = 1
    max_rank = 30 if datetime.date.today().weekday() <= 4 else 60
    article_divs = [f"<p>Today's top {max_rank} stories</p>"]
    rendered_articles = []
    for article in article_list[:max_rank]:
        summary_render = (
            article.summary
            if "".join(re.findall(r"\w", article.title)).lower()
            != "".join(re.findall(r"\w", article.summary)).lower()
            else ""
        )
        div = f"""
            <a href={article.url} class="article-card" target="_blank">
                <div class="article-number">{rank}</div>
                <div class="article-content">
                    <div class="article-title">{article.title}</div>
                    <div class="article-summary">{summary_render}</div>
                </div>
            </a>
            """
        article_divs.append(div)
        rendered_articles.append(article.id)
        rank += 1
    header_html, footer_html = make_base_html()
    article_html = header_html + "\n".join(article_divs) + footer_html
    if test_mode:
        with open("assets/test/render.html", "w") as f:
            f.write(article_html)
    remaining = [i for i in article_list if i.id not in rendered_articles]
    return article_html, remaining
