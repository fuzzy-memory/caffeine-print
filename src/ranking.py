import json
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from openai.types.chat import ChatCompletion
from sklearn.feature_extraction.text import TfidfVectorizer
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(
    wait=wait_exponential(multiplier=2, max=600),
    stop=stop_after_attempt(5),
)
def call_model(prompt: Dict[str, str], client) -> Optional[ChatCompletion]:
    model = "gpt-4o-mini"
    response = client.chat.completions.create(
        model=model, messages=[prompt]  # type: ignore
    )
    return response


def chatgpt_ranking():
    # todo: add sub-category parameters in prompt
    #  leverage structured outputs in GPT response
    news_items = [
        i
        for i in json.load(open("assets/news.json", "r"))
        if all(i.get(k) is not None for k in ["title", "summary", "text"])
        and -0.8 < i.get("sentiment") <= 0.8
    ]
    client = OpenAI()
    total_time = 0
    for q in news_items:
        start = time.time()
        prompt = {
            "role": "user",
            "content": f"""You are helping a news aggregator sift through many news articles. Does the following article report news that has a bearing on Indian polity, Indian economy or global current affairs? Respond with only `true` or `false`. The article is as follows:\n{q.get("text")}""",
        }
        response = call_model(prompt, client)
        if response is None:
            print(f"Unable to retrieve reply after 5. Skipping question: {q.get('id')}")
        raw_response = response.choices[0].message.content
        q.update({"response": raw_response})

        if raw_response.lower().strip() == "true":
            q.update({"score": 9})
        elif raw_response.lower().strip() == "false":
            q.update({"score": 1})
        else:
            q.update({"score": 0.1})
        total_time += time.time() - start
    print()
    print(
        f"Retrieved {len(news_items)} responses in {round(total_time, 4)} seconds or {round(total_time/60, 4)} mins"
    )
    print(
        f"Average time taken per response: {round(total_time / len(news_items), 4)} seconds"
    )

    tfidf = TfidfVectorizer(stop_words="english")
    texts = [i.get("text", "").lower() for i in news_items]

    text_tfidf_matrix = tfidf.fit_transform(texts)

    text_scores = np.mean(text_tfidf_matrix.toarray(), axis=1)
    sentiment_scores = [abs(i.get("sentiment")) for i in news_items]
    source_scores = json.load(open("assets/news_sources.json", "r"))

    # Compute final relevance
    weights = {"source": 0.1, "sentiment": 0.05, "text": 0.2, "score": 0.65}
    relevance_scores = []
    for i, article in enumerate(news_items):
        source_score = source_scores.get(
            "https://" + article["url"].replace("https://", "").split("/")[0]
        )
        score = (
            weights["source"] * source_score
            + weights["sentiment"] * sentiment_scores[i] * 100
            + weights["score"] * article["score"]
            + weights["text"] * text_scores[i] * 100
        )
        relevance_scores.append(score)
    # Add scores to news items and sort
    for i, article in enumerate(news_items):
        article["relevance_score"] = relevance_scores[i]
    sorted_articles = sorted(
        news_items, key=lambda x: x["relevance_score"], reverse=True
    )
    df = pd.DataFrame(sorted_articles)
    return df
