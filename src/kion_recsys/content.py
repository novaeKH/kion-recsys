"""Metadata cosine similarity fitted only on the historical catalog."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from kion_recsys.data import TrainingData
from kion_recsys.models import ItemKNNRecommender, sparse_neighbors


class ContentRecommender(ItemKNNRecommender):
    def __init__(self, neighbors: int = 100):
        self.neighbors = neighbors

    def fit(
        self, interactions: pd.DataFrame | TrainingData, items: pd.DataFrame
    ) -> "ContentRecommender":
        self._set_data(interactions)
        metadata = items.set_index("item_id").reindex(self.data.item_ids)
        fields = ["genres", "countries", "directors", "content_type"]
        documents = []
        for row in metadata[fields].fillna("").itertuples(index=False, name=None):
            tokens = [
                f"{field}={value.strip().lower().replace(' ', '_')}"
                for field, text in zip(fields, row)
                for value in str(text).split(",")
                if value.strip()
            ]
            documents.append(" ".join(tokens) or "missing_metadata")
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False, dtype=np.float32
        )
        self.item_features = self.vectorizer.fit_transform(documents)
        self.similarity = sparse_neighbors(self.item_features, self.neighbors)
        return self
