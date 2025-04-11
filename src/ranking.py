import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from openai.types.chat import ChatCompletion
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from data_models import Article, ArticleScoreMetrics, GPTArticleEvaluationMetrics
from properties import gpt_category_multipliers, testing_gpt_threshold
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
    total_time = 0.0
    scored_articles: List[Article] = []
    print("Sending API calls to ChatGPT")
    for article in tqdm(news):
        start = time.time()
        if article.api_query_category == "laurels":
            article.gpt_feedback = GPTArticleEvaluationMetrics(
                indian_polity=0,
                indian_economy=0,
                indian_local_news=0,
                global_current_affairs=0,
                geopolitics=0,
                entertainment=0,
            )
            scored_articles.append(article)
            total_time += time.time() - start
            continue
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
        if not raw_response:
            raise ValueError(
                f"No response received from ChatGPT API for article ID {article.id}"
            )
        if not isinstance(raw_response, str):
            raise TypeError(
                f"Response of unknown type returned by ChatGPT API: {type(raw_response)}"
            )
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


def softmax(metrics: Dict[str, float]) -> Dict[str, np.float64]:
    metric_arr = np.array(list(metrics.values()))
    exp_arr = np.exp(metric_arr)
    softmax_vals = exp_arr / np.sum(exp_arr)
    return dict(zip(metrics.keys(), softmax_vals))


def calculate_gpt_weighted_score(
    metrics: GPTArticleEvaluationMetrics,
) -> Tuple[float, str]:
    metric_dict = vars(metrics)
    assert set(metric_dict.keys()) == set(gpt_category_multipliers.keys())
    softmax_metrics = softmax(metric_dict)
    highest_scoring_metric = max(softmax_metrics, key=softmax_metrics.get)  # type: ignore
    highest_scoring_value = max(softmax_metrics.values())
    multiplier = gpt_category_multipliers.get(highest_scoring_metric)
    if multiplier is None:
        raise ValueError(
            f"Unable to find multiplier for metric {highest_scoring_metric}"
        )

    final_score = multiplier * highest_scoring_value
    if highest_scoring_metric in ["geopolitics", "global_current_affairs"]:
        tag = "international"
    elif highest_scoring_metric.startswith("indian"):
        tag = "national"
    elif highest_scoring_metric == "entertainment":
        tag = "entertainment"
    else:
        raise ValueError("Could not resolve appropriate tag")
    return final_score, tag


def rank_articles(news_items: List[Article], test_mode: bool):
    if test_mode:
        dir_path = "assets/test/"
        pth = os.path.join(os.path.curdir, dir_path)
        if "chatgpt_ranking.json" in os.listdir(pth):
            df = pd.read_json("assets/test/chatgpt_ranking.json")
            print(f"{df.shape[0]} ranked articles already exist in `assets/test`")
            return df
        else:
            if not testing_gpt_threshold:
                final_threshold = len(news_items)
            else:
                final_threshold = min(len(news_items), testing_gpt_threshold)
        news_items = news_items[:final_threshold]
        print(f"Limiting to {len(news_items)} articles")

    gpt_scored_articles = rank_via_chatgpt(news=news_items)
    print(f"Generated GPT scoring for {len(gpt_scored_articles)} articles")

    text_scores = run_tfidf(gpt_processed_articles=gpt_scored_articles)
    print(f"Generated TF-IDF scoring for {len(text_scores)} articles")
    sentiment_scores = [abs(i.sentiment) for i in gpt_scored_articles]
    source_scores = json.load(open("assets/control.json", "r")).get("source_scores")

    # Compute final relevance
    relevance_scored_articles = []
    for i, article in enumerate(gpt_scored_articles):
        source_score = source_scores.get(article.source, 0)
        chat_gpt_weighted_score, tag = calculate_gpt_weighted_score(
            article.gpt_feedback
        )
        article.relevance_trace = ArticleScoreMetrics(
            source_score=source_score,
            sentiment_score=sentiment_scores[i],
            gpt_feedback_score=chat_gpt_weighted_score,
            text_score=text_scores[i],
            cluster_count=article.cluster_count,
        )
        article.relevance_score = article.relevance_trace.relevance
        article.tag = (
            "awards_and_laurels" if article.api_query_category == "laurels" else tag
        )
        relevance_scored_articles.append(article)
    print(f"Calculated {len(relevance_scored_articles)} relevance scores")

    relevance_scored_articles = [
        i
        for i in relevance_scored_articles
        if i.gpt_feedback.entertainment < 4 and i.tag != "entertainment"
    ]

    print(f"Final scored news articles: {len(relevance_scored_articles)}")
    df = pd.DataFrame([i.to_json() for i in relevance_scored_articles])
    df.sort_values(
        by="relevance_score",
        ascending=False,
        inplace=True,
        ignore_index=True,
    )
    if test_mode:
        df.to_json("assets/test/chatgpt_ranking.json", orient="records")
    return df
