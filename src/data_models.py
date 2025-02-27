from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from properties import sentiment_threshold


class GPTResponse(BaseModel):
    response: str
    score: float

    @model_validator(mode="after")
    def __validate_gpt_response(self):
        self.response = self.response.strip().lower()
        if self.response == "true":
            self.score = 9
        elif self.response == "false":
            self.score = 1
        else:
            self.score = 0.1
        return self

    def __str__(self):
        return f"[Response: {self.response} | Score: {self.score}]"


class Article(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: int
    title: str
    text: str
    summary: str
    url: str
    sentiment: float
    source: Optional[str] = None
    gpt_feedback: Optional[GPTResponse] = None
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
