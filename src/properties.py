sentiment_threshold = 0.75
dbscan_epsilon = 0.1
testing_gpt_threshold = 5
overall_weights = {
    "source": 0.05,
    "sentiment": 0.05,
    "text": 0.15,
    "score": 0.75,
}
gpt_category_multipliers = {
    "indian_polity": 0.5,
    "indian_economy": 0.2,
    "geopolitics": 0.125,
    "global_current_affairs": 0.125,
    "indian_local_news": 0.05,
}

assert sum(gpt_category_multipliers.values()) == 1.0
