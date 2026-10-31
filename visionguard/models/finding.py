from typing import List, Literal
from pydantic import BaseModel, Field

class Finding(BaseModel):
    rule_id: str
    severity: Literal["error", "warning", "info"]
    message: str
    refs: List[str] = Field(default_factory=list)

    model_config = {
        "extra": "ignore"
    }

