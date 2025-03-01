import json
import string
from typing import List

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


def pick_article_from_cluster(articles: List[Article]):
    source_scores = json.load(open("assets/news_sources.json", "r"))

    clustered_articles = {}
    for article in articles:
        cluster_id = article.dbscan_cluster_label
        source = article.source
        source_rank = source_scores.get(source)

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
    # Read news items
    path_to_read = "assets/" + ("test/" if test_mode else "") + "news.json"
    news_items_raw = [
        Article(**i)
        for i in json.load(open(path_to_read, "r"))
        if all(i.get(k) is not None for k in ["title", "summary", "text"])
    ]
    news_items = [i for i in news_items_raw if not i.is_skipped]
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

    deduplicated_articles = pick_article_from_cluster(news_items)
    print(f"Extracted {len(deduplicated_articles)} articles after clustering")
    return deduplicated_articles
