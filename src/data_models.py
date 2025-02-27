from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from properties import sentiment_threshold


class GPTArticleEvaluationModel(BaseModel):
    indian_polity: float
    indian_economy: float
    global_current_affairs: float
    geopolitics: float
    indian_local_news: float

    def __str__(self):
        return f"[Polity: {self.indian_polity} | Economy: {self.indian_economy} | Global CA: {self.global_current_affairs} | Geopol: {self.geopolitics} | Local: {self.indian_local_news}]"


class Article(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: int
    title: str
    text: str
    summary: str
    url: str
    sentiment: float
    source: Optional[str] = None
    gpt_feedback: Optional[GPTArticleEvaluationModel] = None
    relevance_score: Optional[float] = 0.0
    is_skipped: Optional[bool] = False

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
        return self

    def __str__(self):
        st = f"Article #{self.id}: {self.title}"
        prefix = "[SKIPPED] " if self.is_skipped else f"[Rel {self.relevance_score}] "
        suffix = f" | Feedback: {str(self.gpt_feedback)}" if self.gpt_feedback else ""
        return f"{prefix}{st}{suffix}"

    def to_json(self):
        return {
            k: (v if k != "gpt_feedback" else vars(v)) for k, v in vars(self).items()
        }
