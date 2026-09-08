import datetime

sentiment_threshold = 0.75
testing_gpt_threshold = 5
overall_weights = {
    "source": 0.05,
    "sentiment": 0.05,
    "text": 0.05,
    "score": 0.8,
    "cluster_count": 0.05,
}
gpt_category_multipliers = {
    "current_affairs": {
        "indian_polity": 0.65,
        "indian_economy": 0.1,
        "geopolitics": 0.175,
        "global_current_affairs": 0.125,
        "indian_local_news": -0.05,
        "entertainment": 0,
    },
    "business": {
        "bank_loan_fraud": 1 / 13,
        "forgery": 1 / 13,
        "loan_fraud": 1 / 13,
        "gold_loan_fraud": 1 / 13,
        "cyber_fraud": 1 / 13,
        "digital_arrest": 1 / 13,
        "employee_fraud": 1 / 13,
        "insolvency_resolution_fraud": 1 / 13,
        "vendor_fraud": 1 / 13,
        "treasury_fraud": 1 / 13,
        "accounting_fraud": 1 / 13,
        "investment_fraud": 1 / 13,
        "job_fraud": 1 / 13,
    },
}
assert all(
    sum(gpt_category_multipliers.get(channel).values()) == 1.0
    for channel in gpt_category_multipliers
)

is_weekday = datetime.date.today().weekday() <= 4
total_news_items = 30 if is_weekday else 70
international_news_items = 10 if is_weekday else 25
national_news_items = 20 if is_weekday else 35
literature_and_laurels = 0 if is_weekday else 10
assert (
    national_news_items + international_news_items + literature_and_laurels
    == total_news_items
)

permitted_tags = {
    "national": national_news_items,
    "international": international_news_items,
    "awards_and_laurels": literature_and_laurels,
}
chatgpt_model = "gpt-4.1-nano-2025-04-14"
