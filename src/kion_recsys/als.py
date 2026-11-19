"""Implicit matrix factorization: weighted ALS and pairwise BPR."""

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

from kion_recsys.data import TrainingData
from kion_recsys.models import MatrixRecommender


class ALSRecommender(MatrixRecommender):
    def __init__(
        self,
        factors: int = 64,
        confidence: float = 20.0,
        regularization: float = 0.05,
        iterations: int = 15,
        random_state: int = 42,
    ):
        self.factors = factors
        self.confidence = confidence
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, interactions: pd.DataFrame | TrainingData) -> "ALSRecommender":
        self._set_data(interactions)
        confidence = self.data.user_item.copy()
        confidence.data[:] = 1.0 + self.confidence
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            alpha=1.0,
            use_gpu=False,
            num_threads=1,
        )
        self.model.fit(confidence, show_progress=False)
        return self

    def score(self, positions: np.ndarray) -> np.ndarray:
        return self.model.user_factors[positions] @ self.model.item_factors.T


class BPRRecommender(MatrixRecommender):
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        learning_rate: float = 0.05,
        iterations: int = 50,
        random_state: int = 42,
    ):
        self.factors = factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, interactions: pd.DataFrame | TrainingData) -> "BPRRecommender":
        self._set_data(interactions)
        self.model = BayesianPersonalizedRanking(
            factors=self.factors,
            regularization=self.regularization,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=False,
            num_threads=1,
            verify_negative_samples=True,
        )
        self.model.fit(self.data.user_item, show_progress=False)
        return self

    def score(self, positions: np.ndarray) -> np.ndarray:
        return self.model.user_factors[positions] @ self.model.item_factors.T
