"""Macro-averaged ranking metrics over explicit user target sets."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def build_targets(
    history: pd.DataFrame, future: pd.DataFrame, include_cold: bool = True
) -> dict[int, set[int]]:
    """Exclude every previously observed pair; retain unseen items as misses."""
    seen = history.groupby("user_id")["item_id"].agg(set)
    targets = {}
    for user_id, items in future.groupby("user_id")["item_id"].agg(set).items():
        if not include_cold and user_id not in seen:
            continue
        relevant = items - seen.get(user_id, set())
        if relevant:
            targets[int(user_id)] = relevant
    return targets


def per_user_metrics(
    recommendations: Mapping[int, Sequence[int]],
    targets: Mapping[int, set[int]],
    k: int = 10,
) -> pd.DataFrame:
    if k <= 0 or not targets or any(not items for items in targets.values()):
        raise ValueError("Evaluation requires k > 0 and nonempty target sets")
    discounts = 1 / np.log2(np.arange(2, k + 2))
    rows = []
    for user_id, relevant in targets.items():
        predicted = list(recommendations.get(user_id, []))[:k]
        if len(predicted) != len(set(predicted)):
            raise ValueError(f"Duplicate recommendations for user {user_id}")
        hits = np.array([item in relevant for item in predicted], dtype=float)
        precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        ideal_hits = min(len(relevant), k)
        rows.append(
            (
                user_id,
                (precision * hits).sum() / ideal_hits,
                hits.sum() / len(relevant),
                hits @ discounts[: len(hits)] / discounts[:ideal_hits].sum(),
                float(hits.any()),
            )
        )
    return pd.DataFrame(
        rows,
        columns=["user_id", f"MAP@{k}", f"Recall@{k}", f"NDCG@{k}", f"HitRate@{k}"],
    ).set_index("user_id")


def ranking_metrics(
    recommendations: Mapping[int, Sequence[int]],
    targets: Mapping[int, set[int]],
    catalog_size: int,
    k: int = 10,
) -> dict[str, float | int]:
    if catalog_size <= 0:
        raise ValueError("catalog_size must be positive")
    per_user = per_user_metrics(recommendations, targets, k)
    recommended = {
        item for user in targets for item in list(recommendations.get(user, []))[:k]
    }
    return {
        **per_user.mean().to_dict(),
        "Coverage": len(recommended) / catalog_size,
        "Users": len(targets),
    }


def candidate_recall(
    candidates: pd.DataFrame, targets: Mapping[int, set[int]]
) -> float:
    retrieved = candidates.groupby("user_id")["item_id"].agg(set).to_dict()
    return float(
        np.mean(
            [
                len(relevant & retrieved.get(user, set())) / len(relevant)
                for user, relevant in targets.items()
            ]
        )
    )
