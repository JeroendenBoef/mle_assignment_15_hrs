import numpy as np
from typing import List
from pydantic import BaseModel


class PreprocessResponse(BaseModel):
    features: list

    class Config:
        schema_extra = {"example": {"features": ""}}


class InferResponse(BaseModel):
    features: list
    preprocess: bool = False

    class Config:
        schema_extra = {"example": {"features": ""}}
