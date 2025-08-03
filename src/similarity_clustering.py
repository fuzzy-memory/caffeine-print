import json
import string
from typing import Dict, List

import pandas as pd
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

from data_models import Article
from properties import dbscan_epsilon


def preprocess_text(text: str):
    english_stopwords = set(stopwords.words("english"))
    text_translated = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(
        [word for word in text_translated.split() if word not in english_stopwords]
    )


def pick_article_from_cluster(articles: List[Article], source_scores: Dict[str, int]):
    clustered_articles = {}
    for article in articles:
        cluster_id = article.dbscan_cluster_label
        source = article.source
        source_rank = source_scores.get(source)
        if source_rank is None:
            article.is_skipped = True
            continue

        if cluster_id not in clustered_articles:
            clustered_articles.update({cluster_id: article})
        else:
            existing_article = clustered_articles.get(cluster_id)
            existing_source_rank = source_scores.get(existing_article.source)

            # Prefer the article with the higher source rank
            if source_rank > existing_source_rank:
                clustered_articles.update({cluster_id: article})
            # If the source rank is the same, prefer the longer summary
            elif source_rank == existing_source_rank and len(
                article.bert_processed_text
            ) > len(existing_article.bert_processed_text):
                clustered_articles.update({cluster_id: article})

    return list(clustered_articles.values())


def deduplicate_articles(test_mode: bool = False):
    control_json = json.load(open("assets/control.json", "r"))
    source_scores_json = control_json.get("source_scores")
    negative_filters = control_json.get("negative_filters")

    # Read news items
    path_to_read = "assets/" + ("test/" if test_mode else "") + "news.json"
    news_items_raw = [
        Article(**i)
        for i in json.load(open(path_to_read, "r"))
        if all(i.get(k) is not None for k in ["title", "summary", "text"])
    ]
    news_items = []
    for raw_art in news_items_raw:
        top_categories = set(i.strip("-") for i in [
            x.strip()
            for x in raw_art.url.replace(raw_art.source, "").split("/")
            if x.strip() != ""
        ][:4] if i.strip("-").count("-")<=2)

        if (
            all(i not in negative_filters for i in top_categories)
            and (all(x not in raw_art.url for x in ["videoshow", "nyt-connections"]))
            and (not raw_art.is_skipped)
        ):
            news_items.append(raw_art)
    print(f"Parsed {len(news_items)} articles from JSON")

    # Load SBERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Process article titles and text
    for article in news_items:
        article.bert_processed_text = preprocess_text(
            article.title + " " + article.text
        )

    sentences = [article.bert_processed_text for article in news_items]
    embeddings = model.encode(sentences, normalize_embeddings=True)

    # Cluster using DBSCAN
    clustering = DBSCAN(eps=dbscan_epsilon, min_samples=1, metric="cosine").fit(
        embeddings
    )

    # Assign cluster labels back to articles
    for i, article in enumerate(news_items):
        article.dbscan_cluster_label = clustering.labels_[i]

    cluster_counts = {
        int(cluster_label): len(
            [i for i in news_items if i.dbscan_cluster_label == cluster_label]
        )
        for cluster_label in set(i.dbscan_cluster_label for i in news_items)
    }
    for article in news_items:
        article.cluster_count = cluster_counts.get(article.dbscan_cluster_label)
        if article.source not in source_scores_json.keys():
            news_items.pop(news_items.index(article))

    if test_mode:
        pd.DataFrame(
            [
                {
                    k: v
                    for k, v in article.to_json().items()
                    if k != "bert_processed_text"
                }
                for article in news_items
            ]
        ).to_excel("assets/test/similarity_clustering.xlsx", index=False)

    deduplicated_articles = pick_article_from_cluster(news_items, source_scores_json)
    print(f"Extracted {len(deduplicated_articles)} articles after clustering")
    return deduplicated_articles
