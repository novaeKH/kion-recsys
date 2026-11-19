"""Print personalized recommendations from a reproducible historical snapshot."""

import argparse
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from kion_recsys.data import (
    TEST_START,
    TrainingData,
    load_interactions,
    positive_interactions,
)
from kion_recsys.experiment import fit_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument(
        "--model",
        choices=["Popularity", "ItemKNN", "ALS", "BPR", "Content"],
        default="ALS",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("reports/full"))
    args = parser.parse_args()
    selected = json.loads((args.results / "selected_params.json").read_text())
    config = json.loads((args.results / "config.json").read_text())
    history = load_interactions(args.data_dir)
    history = history.loc[history.last_watch_dt < TEST_START]
    positive = positive_interactions(history, config["min_watched_pct"])
    items = pd.read_csv(args.data_dir / "items.csv")
    data = TrainingData.from_frame(positive, history)
    with threadpool_limits(limits=1, user_api="blas"):
        model = fit_model(args.model, selected[args.model], data, items, config["seed"])
        recommendations = model.recommend([args.user_id], config["k"])[args.user_id]
    recent = (
        history.loc[history.user_id.eq(args.user_id)]
        .sort_values("last_watch_dt")
        .tail(10)
    )
    print(f"History before {TEST_START.date()}, user {args.user_id}:")
    print(
        recent.merge(items[["item_id", "title"]], on="item_id")[
            ["title", "last_watch_dt"]
        ].to_string(index=False)
    )
    print(f"\n{args.model} recommendations:")
    result = items.set_index("item_id").loc[recommendations, ["title", "genres"]]
    print(result.to_string())


if __name__ == "__main__":
    main()
