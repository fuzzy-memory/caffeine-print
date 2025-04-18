from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
)

from properties import (
    overall_weights,
    permitted_callers,
    permitted_tags,
    sentiment_threshold,
)


class GPTArticleEvaluationMetrics(BaseModel):
    indian_polity: float
    indian_economy: float
    global_current_affairs: float
    geopolitics: float
    indian_local_news: float
    entertainment: float

    @model_validator(mode="after")
    def __validate_metrics(self):
        for attr in vars(self).keys():
            setattr(self, attr, max(0.0, getattr(self, attr)))
            setattr(self, attr, min(10.0, getattr(self, attr)))
        return self

    def __str__(self):
        return f"[Polity: {self.indian_polity} | Economy: {self.indian_economy} | Global CA: {self.global_current_affairs} | Geopol: {self.geopolitics} | Local: {self.indian_local_news} | Entertainment: {self.entertainment}]"


class ArticleScoreMetrics(BaseModel):
    source_score: float
    sentiment_score: float
    gpt_feedback_score: float
    text_score: float
    cluster_count: float

    @computed_field
    @property
    def relevance(self) -> float:
        return (
            self.source_score * overall_weights.get("source")
            + self.sentiment_score * overall_weights.get("sentiment") * 10
            + self.gpt_feedback_score * overall_weights.get("score") * 10
            + self.text_score * overall_weights.get("text") * 100
            + self.cluster_count * overall_weights.get("cluster_count")
        )

    def to_json(self):
        return {
            "source_score": self.source_score,
            "sentiment_score": self.sentiment_score,
            "gpt_feedback_score": self.gpt_feedback_score,
            "text_score": self.text_score,
            "cluster_count": self.cluster_count,
        }

    def __str__(self):
        return f"[Source: {self.source_score} | Sentiment: {self.sentiment_score} | GPT: {self.gpt_feedback_score} | TF-IDF: {self.text_score} | Count: {self.cluster_count} | Relevance: {self.relevance}]"


class Article(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: int
    title: str
    text: str
    summary: str
    url: str
    sentiment: float
    source: Optional[str] = None
    gpt_feedback: Optional[GPTArticleEvaluationMetrics] = None
    relevance_trace: Optional[ArticleScoreMetrics] = None
    relevance_score: Optional[float] = 0.0
    is_skipped: Optional[bool] = False
    bert_processed_text: Optional[str] = None
    dbscan_cluster_label: Optional[int] = None
    tag: Optional[str] = None
    api_query_category: Optional[str] = None
    cluster_count: Optional[int] = None

    @model_validator(mode="after")
    def __validate_article(self: "Article") -> "Article":
        self.title = self.title.strip()
        self.text = self.text.strip()
        self.summary = self.summary.strip()
        self.source = "https://" + self.url.replace("https://", "").split("/")[0]

        self.is_skipped = (
            any(len(i) == 0 for i in [self.text, self.summary, self.title])
            or abs(self.sentiment) >= sentiment_threshold
        )
        self.cluster_count = 0 if not self.cluster_count else self.cluster_count
        return self

    @field_validator("tag")
    def __validate_tag(cls, tag_val):
        if tag_val not in permitted_tags.keys():
            raise ValueError(
                f"Tag {tag_val} not permitted. Permitted tags are {', '.join(permitted_tags.keys())}"
            )
        return tag_val

    @field_validator("api_query_category")
    def __validate_api_query_category(cls, val):
        if val not in permitted_callers:
            raise ValueError(
                f"Caller {val} not permitted. Permitted callers are {', '.join(permitted_callers)}"
            )
        return val

    def __str__(self):
        st = f"Article #{self.id}: {self.title}"
        prefix = "[SKIPPED] " if self.is_skipped else f"[Rel {self.relevance_score}] "
        suffix = f" | Feedback: {str(self.gpt_feedback)}" if self.gpt_feedback else ""
        return f"{prefix}{st}{suffix}"

    def to_json(self):
        base_dict = {
            k: v
            for k, v in vars(self).items()
            if k not in ["gpt_feedback", "bert_processed_text", "relevance_trace"]
        }
        if self.gpt_feedback:
            base_dict.update(vars(self.gpt_feedback))
        if self.relevance_trace:
            base_dict.update(self.relevance_trace.to_json())
        return base_dict
