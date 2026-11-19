"""Candidate union, reciprocal rank fusion and time-safe ranking features."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from kion_recsys.models import MatrixRecommender


def generate_candidates(
    models: Mapping[str, MatrixRecommender], users, depth: int = 30
) -> pd.DataFrame:
    """Outer union of model lists, with missing ranks encoded as zero evidence."""
    users = list(users)
    frames = []
    for name, model in models.items():
        recommendations = model.recommend(users, k=depth)
        frame = pd.DataFrame(
            [
                (user, item, 1.0 / (60 + rank))
                for user, items in recommendations.items()
                for rank, item in enumerate(items, start=1)
            ],
            columns=["user_id", "item_id", f"rr_{name}"],
        ).astype({"user_id": "int32", "item_id": "int32", f"rr_{name}": "float32"})
        frames.append(frame.set_index(["user_id", "item_id"]))
    return pd.concat(frames, axis=1).fillna(0).reset_index()


def recommendations_from_frame(
    frame: pd.DataFrame, score: str, k: int = 10
) -> dict[int, list[int]]:
    ordered = (
        frame.sort_values(
            ["user_id", score, "item_id"], ascending=[True, False, True], kind="stable"
        )
        .groupby("user_id", sort=False)
        .head(k)
    )
    return ordered.groupby("user_id", sort=False)["item_id"].agg(list).to_dict()


def hybrid_recommendations(
    candidates: pd.DataFrame, weights: Mapping[str, float], k: int = 10
) -> dict[int, list[int]]:
    scores = sum(candidates[f"rr_{name}"] * weight for name, weight in weights.items())
    return recommendations_from_frame(candidates.assign(score=scores), "score", k)


def add_labels(candidates: pd.DataFrame, targets: dict[int, set[int]]) -> pd.DataFrame:
    result = candidates.copy()
    result["target"] = np.fromiter(
        (
            item in targets.get(user, set())
            for user, item in zip(result.user_id, result.item_id)
        ),
        dtype=np.int8,
        count=len(result),
    )
    return result


def ranking_features(
    candidates: pd.DataFrame,
    positive_history: pd.DataFrame,
    items: pd.DataFrame,
    cutoff: pd.Timestamp,
    als: MatrixRecommender,
) -> pd.DataFrame:
    """Only pre-cutoff aggregates and static metadata enter the feature table."""
    if positive_history["last_watch_dt"].max() >= cutoff:
        raise ValueError("Ranking history must precede the prediction cutoff")
    frame = candidates.copy()
    user_counts = positive_history.groupby("user_id")["item_id"].nunique()
    item_counts = positive_history.groupby("item_id")["user_id"].nunique()
    recent = (
        positive_history.loc[
            positive_history.last_watch_dt >= cutoff - pd.Timedelta(days=7)
        ]
        .groupby("item_id")["user_id"]
        .nunique()
    )
    last_seen = positive_history.groupby("user_id")["last_watch_dt"].max()
    frame["user_activity"] = np.log1p(frame.user_id.map(user_counts).fillna(0))
    frame["item_popularity"] = np.log1p(frame.item_id.map(item_counts).fillna(0))
    frame["item_recent_popularity"] = np.log1p(frame.item_id.map(recent).fillna(0))
    frame["user_recency_days"] = (cutoff - frame.user_id.map(last_seen)).dt.days.fillna(
        365
    )
    frame["source_count"] = frame.filter(like="rr_").gt(0).sum(axis=1)
    frame["rrf_score"] = frame.filter(like="rr_").sum(axis=1)
    metadata = items.set_index("item_id")
    frame["release_year"] = frame.item_id.map(metadata.release_year).fillna(0)
    frame["age_rating"] = frame.item_id.map(metadata.age_rating).fillna(0)
    frame["is_series"] = frame.item_id.map(metadata.content_type).eq("series")
    user_positions = als.data.user_index.get_indexer(frame.user_id)
    item_positions = pd.Index(als.data.item_ids).get_indexer(frame.item_id)
    values = np.zeros(len(frame), dtype=np.float32)
    # Chunk pair scoring to avoid allocating a candidate-count x factors array.
    for start in range(0, len(frame), 100_000):
        stop = min(start + 100_000, len(frame))
        users, content = user_positions[start:stop], item_positions[start:stop]
        valid = (users >= 0) & (content >= 0)
        values[start:stop][valid] = np.einsum(
            "ij,ij->i",
            als.model.user_factors[users[valid]],
            als.model.item_factors[content[valid]],
        )
    frame["als_score"] = values
    feature_columns = frame.columns.difference(["user_id", "item_id", "target"])
    frame[feature_columns] = frame[feature_columns].astype("float32")
    return frame


class CandidateRanker:
    def __init__(self, iterations: int = 150, depth: int = 6, random_state: int = 42):
        self.model = CatBoostRanker(
            iterations=iterations,
            depth=depth,
            learning_rate=0.08,
            loss_function="PairLogit",
            random_seed=random_state,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(
        self, frame: pd.DataFrame, features: list[str] | None = None
    ) -> "CandidateRanker":
        self.features = (
            features
            if features is not None
            else [
                column
                for column in frame.columns
                if column not in {"user_id", "item_id", "target", "query_id"}
            ]
        )
        group = "query_id" if "query_id" in frame else "user_id"
        # Constant-label queries contribute no ranking pairs.
        useful = frame.groupby(group)["target"].transform("nunique").gt(1)
        train = frame.loc[useful].sort_values(group, kind="stable")
        if train.empty:
            raise ValueError(
                "No training queries with both positive and negative candidates"
            )
        self.training_rows = len(train)
        self.training_queries = train[group].nunique()
        self.model.fit(Pool(train[self.features], train.target, group_id=train[group]))
        return self

    def recommend(self, frame: pd.DataFrame, k: int = 10) -> dict[int, list[int]]:
        scored = frame[["user_id", "item_id"]].copy()
        scored["score"] = self.model.predict(frame[self.features])
        return recommendations_from_frame(scored, "score", k)
