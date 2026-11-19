"""Temporal splits and a shared sparse representation of implicit feedback."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

VALIDATION_START = pd.Timestamp("2021-08-09")
TEST_START = pd.Timestamp("2021-08-16")
TEST_END = pd.Timestamp("2021-08-23")
RANKER_START = pd.Timestamp("2021-08-02")


def load_interactions(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        Path(data_dir) / "interactions.csv",
        parse_dates=["last_watch_dt"],
        dtype={"user_id": "int32", "item_id": "int32", "watched_pct": "float32"},
    )


def positive_interactions(
    interactions: pd.DataFrame, min_watched_pct: int = 20
) -> pd.DataFrame:
    """One binary positive per user/item; missing percentages are not positives."""
    return (
        interactions.loc[interactions["watched_pct"].ge(min_watched_pct)]
        .sort_values("last_watch_dt", kind="stable")
        .drop_duplicates(["user_id", "item_id"], keep="last")
    )


def split_interactions(
    interactions: pd.DataFrame, min_watched_pct: int | None = 20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split before deduplication: future repeats cannot erase training history."""
    dates = interactions["last_watch_dt"]
    frames = (
        interactions.loc[dates < VALIDATION_START],
        interactions.loc[(dates >= VALIDATION_START) & (dates < TEST_START)],
        interactions.loc[(dates >= TEST_START) & (dates < TEST_END)],
    )
    if min_watched_pct is None:
        return frames
    return tuple(positive_interactions(frame, min_watched_pct) for frame in frames)


@dataclass
class TrainingData:
    """Shared ID mapping, positive matrix and all observed pairs for filtering."""

    user_index: pd.Index
    item_ids: np.ndarray
    user_item: sparse.csr_matrix
    seen: sparse.csr_matrix
    popular_indices: np.ndarray

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @classmethod
    def from_frame(
        cls, positive: pd.DataFrame, history: pd.DataFrame | None = None
    ) -> "TrainingData":
        if positive.empty:
            raise ValueError("Training requires at least one positive interaction")
        history = positive if history is None else history
        user_index = pd.Index(np.sort(history["user_id"].unique()))
        item_ids = np.sort(positive["item_id"].unique())
        item_index = pd.Index(item_ids)

        def encode(frame: pd.DataFrame) -> sparse.csr_matrix:
            rows = user_index.get_indexer(frame["user_id"])
            columns = item_index.get_indexer(frame["item_id"])
            valid = (rows >= 0) & (columns >= 0)
            matrix = sparse.csr_matrix(
                (np.ones(valid.sum(), dtype=np.float32), (rows[valid], columns[valid])),
                shape=(len(user_index), len(item_ids)),
            )
            matrix.data[:] = 1.0
            return matrix

        user_item = encode(positive)
        counts = user_item.getnnz(axis=0)
        return cls(
            user_index,
            item_ids,
            user_item,
            encode(history),
            np.argsort(-counts, kind="stable"),
        )


def as_training_data(interactions: pd.DataFrame | TrainingData) -> TrainingData:
    if isinstance(interactions, TrainingData):
        return interactions
    return TrainingData.from_frame(interactions)


def cold_start_stats(
    history: pd.DataFrame, future: pd.DataFrame
) -> dict[str, int | float]:
    new_users = ~future["user_id"].isin(history["user_id"])
    new_items = ~future["item_id"].isin(history["item_id"])
    return {
        "future_users": int(future["user_id"].nunique()),
        "new_users": int(future.loc[new_users, "user_id"].nunique()),
        "events_from_new_users_pct": float(new_users.mean() * 100),
        "new_items": int(future.loc[new_items, "item_id"].nunique()),
        "events_for_new_items_pct": float(new_items.mean() * 100),
    }
