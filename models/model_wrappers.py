from abc import ABC, abstractmethod
import lightgbm as lgb
import numpy as np
from typing import List
import pandas as pd

from models.utils import clean_df, convert_to_unix_timedelta
from models.constants import RELEVANT_COLUMNS, ALL_FEATURE_COLS


class GBDTModelABC(ABC):
    @abstractmethod
    def infer(self, input: pd.DataFrame) -> List[int]:
        NotImplementedError()

    @staticmethod
    def chunk_batch(batch, n):
        for i in range(0, len(batch), n):
            yield batch[i : i + n]


class GBDTModel(GBDTModelABC):
    _model = None

    def __init__(
        self,
        batch_size: int = 512,
        model_path: str = "./model_checkpoints/lgb_gbdt_model.pkl",
    ):
        self.batch_size = batch_size
        self.model_path = model_path

    @property
    def model(self):
        if self._model is None:
            print("Loading model")
            self._model = lgb.Booster(model_file=self.model_path)
        return self._model

    def preprocess(self, features: list):
        df = pd.DataFrame(np.array(features), columns=ALL_FEATURE_COLS)
        df = clean_df(df[RELEVANT_COLUMNS].copy())
        df["date_recorded"] = convert_to_unix_timedelta(df)
        return df.values

    def infer(self, features: list, preprocess=False, batching=True) -> List[int]:
        if preprocess:
            features = self.preprocess(features)

        if batching:
            output = []
            for batch in GBDTModel.chunk_batch(features, self.batch_size):
                logits = self.model.predict(batch)
                y_pred = np.argmax(logits, axis=1)
                output.extend(y_pred)
        else:
            features = [features, features]
            logits = self.model.predict(features)
            output = np.argmax(logits, axis=1)[0]

        return output
