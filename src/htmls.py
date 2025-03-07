import datetime
import json
import re
from statistics import mean
from typing import List, Tuple

import pandas as pd
from titlecase import titlecase

from data_models import Article
from properties import permitted_tags


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
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100..900;1,100..900&display=swap"
              rel="stylesheet">
        <title>Hash Brown: {date}</title>
        <style>
            body {{
                background-color: #121212;
                color: white;
                font-family: Roboto, sans-serif;
                margin: 0;
                padding: 10px;
            }}
            .container {{
                width: 80vw;
                margin: auto;
            }}
            .article-card {{
                display: flex;
                align-items: center;
                background-color: #1e1e1e;
                padding: 35px;
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
                padding: 1px;
                display: flex;
                justify-content: center;
                align-items: center;
                width: 50px;
                font-size: 24px;
                font-weight: bold;
                color: #bbbbbb;
                background-color: #292929;
                height: 100%;
                border-radius: 8px;
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
                font-size: 18px;
                padding: 5px;
            }}
            hr.rounded {{
                border-top: 5px solid #bbb;
                border-radius: 5px;
                margin-top: 5px;
            }}
            .article-breaking {{
                padding: 5px;
                font-size: 18px;
                font-weight: bold;
                color: #950606;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Hash Brown: {date}</h1>
    """

    footer_html = """
        <div style="text-align: center;">Made with ❤️ in Mumbai</div>
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
    mean_report_count = max(mean(i.cluster_count for i in article_list), 2)
    article_divs = [f"<p>Today's top stories</p>"]
    rendered_articles = []
    for tag in permitted_tags.keys():
        rank = 1
        max_items = permitted_tags.get(tag)
        if max_items == 0:
            continue
        if len([i for i in article_list if i.tag == tag]) == 0:
            continue
        if tag != "national":
            article_divs.extend(['<hr class="rounded">'])
        article_divs.extend(
            [
                f"<h2>{titlecase(tag.replace('_', ' '))}{' news' if tag.endswith('national') else ''}</h2>"
            ]
        )
        render_items = [i for i in article_list if i.tag == tag][:max_items]
        for article in render_items:
            summary_render = (
                article.summary
                if "".join(re.findall(r"\w", article.title)).lower()
                != "".join(re.findall(r"\w", article.summary)).lower()
                else ""
            )
            div = (
                f"""<a href={article.url} class="article-card" target="_blank">"""
                f"""    <div class="article-number">{rank}</div>"""
                f"""    <div class="article-content">"""
                f"""        <div class="article-title">{article.title}</div>"""
                + (
                    f"""        <div class="article-breaking">BREAKING: Reported {article.cluster_count} times</div>"""
                    if article.cluster_count >= mean_report_count
                    else ""
                )
                + f"""        <div class="article-summary">{summary_render}</div>"""
                f"""    </div>""" + f"""</a>"""
            )
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
