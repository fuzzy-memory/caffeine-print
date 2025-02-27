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
from data_models import Article, GPTResponse

max_openai_retires = 5


@retry(
    wait=wait_exponential(multiplier=2, max=600),
    stop=stop_after_attempt(max_openai_retires),
)
def call_model(prompt: Dict[str, str], client) -> Optional[ChatCompletion]:
    model = "gpt-4o-mini"
    response = client.beta.chat.completions.parse(
        model=model, messages=[prompt], response_format=GPTResponse  # type: ignore
    )
    return response


def rank_via_chatgpt(news: List[Article]):
    client = OpenAI()
    total_time = 0
    scored_articles: List[Article] = []
    for article in news:
        start = time.time()
        prompt = {
            "role": "user",
            "content": (
                f"You are helping a news aggregator sift through many news articles. Does the following "
                f"article report news that has a bearing on Indian polity, Indian economy or global current affairs? "
                f"Respond with only `true` or `false`. The article is as follows:\n{article.text}"
            ),
        }
        response = call_model(prompt, client)
        if response is None:
            print(
                f"Unable to retrieve reply after {max_openai_retires} retries. Skipping question: {article.id}"
            )
        raw_response = response.choices[0].message.content
        article.gpt_feedback = GPTResponse(**json.loads(raw_response))
        scored_articles.append(article)
        total_time += time.time() - start
    print()
    print(
        f"Retrieved {len(news)} responses in {round(total_time, 4)} seconds or {round(total_time / 60, 4)} mins"
    )
    print(
        f"Average time taken per response: {round(total_time / len(news), 4)} seconds"
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
    if test_mode:
        final_threshold=max(len(news_items), testing_gpt_threshold)
        print(f"Limiting to first {final_threshold} articles")
        news_items = news_items[:final_threshold]

    gpt_scored_articles = rank_via_chatgpt(news=news_items)
    text_scores=run_tfidf(gpt_processed_articles=gpt_scored_articles)
    sentiment_scores = [abs(i.sentiment) for i in gpt_scored_articles]
    source_scores = json.load(open("assets/news_sources.json", "r"))

    # Compute final relevance
    weights = {"source": 0.1, "sentiment": 0.05, "text": 0.2, "score": 0.65}
    relevance_scores = []
    for i, article in enumerate(gpt_scored_articles):
        source_score = source_scores.get(article.source)
        score = (
            weights["source"] * source_score
            + weights["sentiment"] * sentiment_scores[i] * 100
            + weights["score"] * article.gpt_feedback.score
            + weights["text"] * text_scores[i] * 100
        )
        relevance_scores.append(score)

    relevance_scored_articles = []
    for i, article in enumerate(gpt_scored_articles):
        article.relevance_score = relevance_scores[i]
        relevance_scored_articles.append(article)
    sorted_articles = sorted(
        relevance_scored_articles, key=lambda x: x.relevance_score, reverse=True
    )
    df = pd.DataFrame([i.to_json() for i in sorted_articles])
    return df
