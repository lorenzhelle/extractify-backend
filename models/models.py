from typing import Union
from pydantic import BaseModel


class FilterGeneratorOutput(BaseModel):
    id: str
    values: list[str]
    minimum: Union[int, None] = None
    maximum: Union[int, None] = None
