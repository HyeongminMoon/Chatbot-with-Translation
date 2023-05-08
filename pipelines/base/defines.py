from pydantic import BaseModel


class DefaultTS():
    ts_tool = "google"
    back_ts_tool = "google"


class PromptData(BaseModel):
    text: str