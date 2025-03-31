from typing import Dict, List


def generate_prompt(article_text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant to a news aggregator sorting through many news articles. Rate the "
            "given articles on the provided metrics as truthfully as possible.",
        },
        {
            "role": "user",
            "content": (
                f"For the given article, evaluate "
                f"its pertinence on the following metrics using softmax probability.\n"
                f"\t1. Indian polity (indian_polity)\n"
                f"\t2. Bearing on the Indian economy (indian_economy)\n"
                f"\t3. Global current affairs (global_current_affairs)\n"
                f"\t4. Geopolitics (geopolitics)\n"
                f"\t5. To what extent does the article report a localised event? Here, 1 would represent extremely local and 10 would represent national level (indian_local_news)\n"
                f"\t6. Movies, music and entertainment\n"
                f"Remember that the probabilities should add up to 1 in a softmax function. The article is as follows:\n{article_text}"
            ),
        },
    ]
