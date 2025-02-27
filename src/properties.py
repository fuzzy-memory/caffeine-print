sentiment_threshold = 0.75
testing_gpt_threshold = 5
overall_weights = {"source": 0.05, "sentiment": 0.05, "text": 0.2, "score": 0.7}
gpt_category_multipliers = {
    "indian_polity": 0.3,
    "indian_economy": 0.3,
    "global_current_affairs": 0.175,
    "geopolitics": 0.175,
    "indian_local_news": 0.05,
}
assert sum([v for k, v in gpt_category_multipliers.items()]) == 1
