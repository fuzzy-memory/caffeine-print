import datetime

sentiment_threshold = 0.75
dbscan_epsilon = 0.25
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

is_weekday = datetime.date.today().weekday() <= 4
total_news_items = 30 if is_weekday else 70
international_news_items = 10 if is_weekday else 20
national_news_items = 20 if is_weekday else 40
literature_and_laurels = 0 if is_weekday else 10
assert national_news_items + international_news_items == total_news_items

permitted_tags = {
    "national": national_news_items,
    "international": international_news_items,
    "awards_and_laurels": literature_and_laurels,
}
permitted_callers = ["general", "laurels"]
