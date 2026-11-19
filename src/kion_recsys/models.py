"""Popularity and sparse cosine ItemKNN with a common serving policy."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from kion_recsys.data import TrainingData, as_training_data


def top_indices(indices: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k with deterministic item-index tie breaking, including the boundary."""
    if len(indices) > k:
        threshold = np.partition(scores, -k)[-k]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        tied = tied[np.argsort(indices[tied])[: k - len(above)]]
        selected = np.concatenate([above, tied])
        indices, scores = indices[selected], scores[selected]
    return indices[np.lexsort((indices, -scores))][:k]


def sparse_neighbors(features: sparse.csr_matrix, neighbors: int) -> sparse.csr_matrix:
    """Blockwise cosine avoids materializing the full item-item similarity."""
    normalized = normalize(features, norm="l2", axis=1).tocsr()
    blocks = []
    for start in range(0, features.shape[0], 256):
        block = (normalized[start : start + 256] @ normalized.T).tocsr()
        rows, columns, values = [], [], []
        for row in range(block.shape[0]):
            left, right = block.indptr[row : row + 2]
            candidates, scores = block.indices[left:right], block.data[left:right]
            valid = (candidates != start + row) & (scores > 0)
            candidates, scores = candidates[valid], scores[valid]
            selected = top_indices(candidates, scores, neighbors)
            score_map = dict(zip(candidates, scores))
            rows.extend([row] * len(selected))
            columns.extend(selected)
            values.extend(score_map[index] for index in selected)
        blocks.append(
            sparse.csr_matrix(
                (np.asarray(values, dtype=np.float32), (rows, columns)),
                shape=(block.shape[0], features.shape[0]),
            )
        )
    return sparse.vstack(blocks, format="csr")


class MatrixRecommender:
    """Shared batching, history filtering and popularity fallback for models."""

    def _set_data(self, interactions: pd.DataFrame | TrainingData) -> None:
        self.data = as_training_data(interactions)
        self.n_items = self.data.n_items
        self.popular_indices = self.data.popular_indices

    def score(self, positions: np.ndarray) -> np.ndarray | sparse.csr_matrix:
        raise NotImplementedError

    def recommend(
        self, users: Iterable[int], k: int = 10, batch_size: int = 256
    ) -> dict[int, list[int]]:
        if k <= 0:
            raise ValueError("k must be positive")
        users = np.asarray(list(users))
        positions = self.data.user_index.get_indexer(users)
        recommendations = {}
        for start in range(0, len(users), batch_size):
            batch_users = users[start : start + batch_size]
            batch_positions = positions[start : start + batch_size]
            known = batch_positions >= 0
            scores = self.score(batch_positions[known]) if known.any() else None
            row = 0
            for user, position in zip(batch_users, batch_positions):
                seen = self.data.seen[position].indices if position >= 0 else []
                selected = []
                if position >= 0:
                    warm = (
                        self.data.user_item.indptr[position + 1]
                        > (self.data.user_item.indptr[position])
                    )
                    if warm:
                        if sparse.issparse(scores):
                            left, right = scores.indptr[row : row + 2]
                            indices = scores.indices[left:right]
                            values = scores.data[left:right]
                            valid = (values > 0) & ~np.isin(indices, seen)
                            indices, values = indices[valid], values[valid]
                        else:
                            values = scores[row].copy()
                            values[seen] = -np.inf
                            indices = np.flatnonzero(np.isfinite(values))
                            values = values[indices]
                        selected = top_indices(indices, values, k).tolist()
                    row += 1
                if len(selected) < k:
                    excluded = set(seen) | set(selected)
                    for index in self.popular_indices:
                        if index not in excluded:
                            selected.append(int(index))
                        if len(selected) == k:
                            break
                recommendations[int(user)] = self.data.item_ids[selected].tolist()
        return recommendations


class PopularRecommender(MatrixRecommender):
    def fit(self, interactions: pd.DataFrame | TrainingData) -> "PopularRecommender":
        self._set_data(interactions)
        self.popular_items = self.data.item_ids[self.data.popular_indices]
        return self

    def score(self, positions: np.ndarray) -> sparse.csr_matrix:
        # An empty score matrix routes every user through the shared fallback.
        return sparse.csr_matrix((len(positions), self.n_items), dtype=np.float32)


class RecentPopularRecommender(PopularRecommender):
    def __init__(self, window_days: int = 7):
        self.window_days = window_days

    def fit(
        self,
        interactions: TrainingData,
        positive_history: pd.DataFrame,
        cutoff: pd.Timestamp,
    ) -> "RecentPopularRecommender":
        self._set_data(interactions)
        recent = positive_history.loc[
            positive_history.last_watch_dt.between(
                cutoff - pd.Timedelta(days=self.window_days), cutoff, inclusive="left"
            )
        ]
        counts = recent.groupby("item_id")["user_id"].nunique()
        recent_counts = counts.reindex(self.data.item_ids, fill_value=0).to_numpy()
        lifetime_counts = self.data.user_item.getnnz(axis=0)
        self.popular_indices = np.lexsort(
            (self.data.item_ids, -lifetime_counts, -recent_counts)
        )
        self.popular_items = self.data.item_ids[self.popular_indices]
        return self


class ItemKNNRecommender(MatrixRecommender):
    def __init__(self, neighbors: int = 50, min_item_interactions: int = 20):
        self.neighbors = neighbors
        self.min_item_interactions = min_item_interactions

    def fit(self, interactions: pd.DataFrame | TrainingData) -> "ItemKNNRecommender":
        self._set_data(interactions)
        counts = self.data.user_item.getnnz(axis=0)
        eligible = np.flatnonzero(counts >= self.min_item_interactions)
        if len(eligible):
            local = sparse_neighbors(
                self.data.user_item[:, eligible].T.tocsr(), self.neighbors
            ).tocoo()
            self.similarity = sparse.csr_matrix(
                (local.data, (eligible[local.row], eligible[local.col])),
                shape=(self.n_items, self.n_items),
            )
        else:
            self.similarity = sparse.csr_matrix((self.n_items, self.n_items))
        return self

    def score(self, positions: np.ndarray) -> sparse.csr_matrix:
        return (self.data.user_item[positions] @ self.similarity).tocsr()
