import json
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from openai.types.chat import ChatCompletion
from sklearn.feature_extraction.text import TfidfVectorizer
from tenacity import retry, stop_after_attempt, wait_exponential

from properties import testing_gpt_threshold
from data_models import Article, GPTArticleEvaluationMetrics
from utils import generate_prompt

max_openai_retires = 5


@retry(
    wait=wait_exponential(multiplier=2, max=600),
    stop=stop_after_attempt(max_openai_retires),
)
def call_model(prompt: List[Dict[str, str]], client) -> Optional[ChatCompletion]:
    model = "gpt-4o-mini"
    response = client.beta.chat.completions.parse(
        model=model, messages=prompt, response_format=GPTArticleEvaluationMetrics  # type: ignore
    )
    return response


def rank_via_chatgpt(news: List[Article]):
    client = OpenAI()
    total_time = 0
    scored_articles: List[Article] = []
    print("Sending API calls to ChatGPT")
    for article in news:
        start = time.time()
        prompt = generate_prompt(article_text=article.text)
        response = call_model(prompt, client)
        if response is None:
            print(
                f"Unable to retrieve reply after {max_openai_retires} retries. Skipping question: {article.id}"
            )
            continue
        if response.choices[0].message.refusal:
            print(
                f"Refusal encountered for {article.id}: {response.choices[0].message.refusal}"
            )
            continue
        raw_response = response.choices[0].message.content
        article.gpt_feedback = GPTArticleEvaluationMetrics(**json.loads(raw_response))
        scored_articles.append(article)
        total_time += time.time() - start
    print()
    print(
        f"Retrieved {len(scored_articles)} responses in {round(total_time, 4)} seconds or {round(total_time / 60, 4)} mins"
    )
    print(
        f"Average time taken per response: {round(total_time / len(scored_articles), 4)} seconds"
    )
    return scored_articles


def run_tfidf(gpt_processed_articles: List[Article]):
    tfidf = TfidfVectorizer(stop_words="english")
    texts = [i.text.lower() for i in gpt_processed_articles]

    text_tfidf_matrix = tfidf.fit_transform(texts)
    text_scores = np.mean(text_tfidf_matrix.toarray(), axis=1)

    return text_scores


def rank_articles(test_mode: bool):
    path_to_read = "assets/" + ("test/" if test_mode else "") + "news.json"
    news_items_raw = [
        Article(**i)
        for i in json.load(open(path_to_read, "r"))
        if all(i.get(k) is not None for k in ["title", "summary", "text"])
    ]
    news_items = [i for i in news_items_raw if not i.is_skipped]
    print(f"Parsed {len(news_items)} articles from JSON")
    if test_mode:
        final_threshold = min(len(news_items), testing_gpt_threshold)
        news_items = news_items[:final_threshold]
        print(f"Limiting to {len(news_items)} articles")

    gpt_scored_articles = rank_via_chatgpt(news=news_items)
    print(f"Generated GPT scoring for {len(gpt_scored_articles)} articles")

    text_scores = run_tfidf(gpt_processed_articles=gpt_scored_articles)
    print(f"Generated TF-IDF scoring for {len(text_scores)} articles")
    sentiment_scores = [abs(i.sentiment) for i in gpt_scored_articles]
    source_scores = json.load(open("assets/news_sources.json", "r"))

    # Compute final relevance
    weights = {"source": 0.05, "sentiment": 0.05, "text": 0.2, "score": 0.7}
    relevance_scores = []
    for i, article in enumerate(gpt_scored_articles):
        source_score = source_scores.get(article.source)
        score = (
            weights["source"] * source_score
            + weights["sentiment"] * sentiment_scores[i] * 10
            + weights["score"] * article.gpt_feedback.score
            + weights["text"] * text_scores[i] * 100
        )
        relevance_scores.append(score)
    print(f"Calculated {len(relevance_scores)} relevance scores")

    relevance_scored_articles = []
    for i, article in enumerate(gpt_scored_articles):
        article.relevance_score = relevance_scores[i]
        relevance_scored_articles.append(article)

    print(f"Final scored news articles: {len(relevance_scored_articles)}")
    df = pd.DataFrame([i.to_json() for i in relevance_scored_articles])
    df.sort_values(
        by="relevance_score",
        ascending=False,
        inplace=True,
        ignore_index=True,
    )
    return df
