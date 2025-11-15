from typing import Literal
from pydantic import BaseModel, Field

class GreetingClassifier(BaseModel):
    result: Literal["yes", "no"] = Field(description="Whether the input is a greeting or not")
