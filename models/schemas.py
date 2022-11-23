from tokenize import Single
import numpy as np
from typing import List
from pydantic import BaseModel


class PreprocessResponse(BaseModel):
    features: list

    class Config:
        schema_extra = {"example": {"features": ""}}


class SingleInferResponse(BaseModel):
    prediction: int


class InferResponse(BaseModel):
    predictions: List[int]

    class Config:
        schema_extra = {"example": {"features": ""}}
