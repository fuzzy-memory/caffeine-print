from typing import List, Dict


def generate_prompt(article_text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant to a news aggregator sorting through many news articles. Guide the user as requested.",
        },
        {
            "role": "user",
            "content": (
                f"For the given article, evaluate"
                f"its pertinence on the following metrics using a scale of 1-10.\n"
                f"\t1. Indian polity (indian_polity)\n"
                f"\t2. Bearing on the Indian economy (indian_economy)\n"
                f"\t3. Global current affairs (global_current_affairs)\n"
                f"\t4. Geopolitics (geopolitics)\n"
                f"\t5. To what extent does the article report a localised event? Here, 1 would represent extremely local and 10 would represent national level (indian_local_news)\n"
                f"The article is as follows:\n{article_text}"
            ),
        },
    ]
