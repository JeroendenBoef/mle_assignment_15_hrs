import numpy as np
from typing import List
from pydantic import BaseModel

# class FeatureArray(BaseModel):
#     features: np.ndarray = None

#     class Config:
#         arbitrary_types_allowed = True


class PreprocessResponse(BaseModel):
    features: list

    class Config:
        # arbitrary_types_allowed = True
        schema_extra = {"example": {"features": ""}}


class InferResponse(BaseModel):
    features: list
    preprocess: bool = False

    class Config:
        # arbitrary_types_allowed = True
        schema_extra = {"example": {"features": ""}}
