"""Run a chronological benchmark and save its results, choices and provenance."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from kion_recsys.als import ALSRecommender, BPRRecommender
from kion_recsys.content import ContentRecommender
from kion_recsys.data import (
    RANKER_START,
    TEST_END,
    TEST_START,
    VALIDATION_START,
    TrainingData,
    load_interactions,
    positive_interactions,
)
from kion_recsys.metrics import (
    build_targets,
    candidate_recall,
    per_user_metrics,
    ranking_metrics,
)
from kion_recsys.models import ItemKNNRecommender, PopularRecommender
from kion_recsys.ranking import (
    CandidateRanker,
    add_labels,
    generate_candidates,
    hybrid_recommendations,
    ranking_features,
)

MODEL_CLASSES = {
    "Popularity": PopularRecommender,
    "ItemKNN": ItemKNNRecommender,
    "ALS": ALSRecommender,
    "BPR": BPRRecommender,
    "Content": ContentRecommender,
}


@dataclass
class EvaluationPeriod:
    name: str
    cutoff: pd.Timestamp
    positive: pd.DataFrame
    targets: dict[int, set[int]]
    data: TrainingData


def make_period(interactions, cutoff, end, name, threshold=20) -> EvaluationPeriod:
    history = interactions.loc[interactions.last_watch_dt < cutoff]
    future = interactions.loc[
        interactions.last_watch_dt.between(cutoff, end, inclusive="left")
    ]
    positive = positive_interactions(history, threshold)
    targets = build_targets(history, positive_interactions(future, threshold))
    return EvaluationPeriod(
        name, cutoff, positive, targets, TrainingData.from_frame(positive, history)
    )


def fit_model(name, params, data, items, seed=42):
    params = dict(params)
    if name in {"ALS", "BPR"}:
        params["random_state"] = seed
    model = MODEL_CLASSES[name](**params)
    return model.fit(data, items) if name == "Content" else model.fit(data)


def save_json(value, path):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n",
        encoding="utf-8",
    )


def append_rows(frame, path):
    frame.to_csv(path, index=False, mode="a", header=not path.exists())


def record_result(output, period, name, recs, seconds, k):
    metrics = ranking_metrics(recs, period.targets, period.data.n_items, k)
    append_rows(
        pd.DataFrame(
            [{"model": name, "split": period.name, **metrics, "seconds": seconds}]
        ),
        output / "metrics.csv",
    )
    scores = per_user_metrics(recs, period.targets, k)
    counts = period.positive.groupby("user_id")["item_id"].nunique()
    scores["segment"] = pd.cut(
        scores.index.to_series().map(counts).fillna(0),
        bins=[-1, 0, 1, 5, 20, np.inf],
        labels=["cold", "1", "2-5", "6-20", "21+"],
    )
    segments = scores.groupby("segment", observed=True).mean()
    segments["Users"] = scores.groupby("segment", observed=True).size()
    segments["model"], segments["split"] = name, period.name
    append_rows(segments.reset_index(), output / "segments.csv")
    print(
        f"{period.name} {name}: MAP={metrics[f'MAP@{k}']:.5f}, "
        f"Recall={metrics[f'Recall@{k}']:.5f}, {seconds:.1f}s",
        flush=True,
    )


def record_grid(output, name, params, metrics, seconds):
    append_rows(
        pd.DataFrame(
            [
                {
                    "model": name,
                    "params": json.dumps(params),
                    **metrics,
                    "seconds": seconds,
                }
            ]
        ),
        output / "validation_grid.csv",
    )


def tune_models(period, items, config, output):
    selected, candidate_models = {}, {}
    for name, grid in config["models"].items():
        best_score = -1.0
        for params in grid:
            tick = time.perf_counter()
            print(f"Fitting validation {name} {params}", flush=True)
            model = fit_model(name, params, period.data, items, config["seed"])
            recs = model.recommend(period.targets, config["k"])
            metrics = ranking_metrics(
                recs, period.targets, period.data.n_items, config["k"]
            )
            seconds = time.perf_counter() - tick
            record_grid(output, name, params, metrics, seconds)
            if params == config["candidate_models"].get(name):
                candidate_models[name] = model
            if metrics[f"MAP@{config['k']}"] > best_score:
                best_score, best_recs, best_seconds = (
                    metrics[f"MAP@{config['k']}"],
                    recs,
                    seconds,
                )
                selected[name] = params
        record_result(output, period, name, best_recs, best_seconds, config["k"])
    return selected, candidate_models


def make_candidates(period, items, config, output, models=None):
    models = {} if models is None else models
    for name, params in config["candidate_models"].items():
        if name not in models:
            models[name] = fit_model(name, params, period.data, items, config["seed"])
    print(f"Generating {period.name} candidates", flush=True)
    tick = time.perf_counter()
    candidates = generate_candidates(models, period.targets, config["candidate_depth"])
    seconds = time.perf_counter() - tick
    append_rows(
        pd.DataFrame(
            [
                {
                    "split": period.name,
                    "rows": len(candidates),
                    "users": len(period.targets),
                    "candidate_recall": candidate_recall(candidates, period.targets),
                }
            ]
        ),
        output / "candidate_recall.csv",
    )
    labelled = (
        candidates if period.name == "test" else add_labels(candidates, period.targets)
    )
    features = ranking_features(
        labelled, period.positive, items, period.cutoff, models["ALS"]
    )
    return features, seconds


def tune_hybrid(features, period, config, output, retrieval_seconds):
    best_score = -1.0
    for weights in config["hybrid_weights"]:
        tick = time.perf_counter()
        recs = hybrid_recommendations(features, weights, config["k"])
        metrics = ranking_metrics(
            recs, period.targets, period.data.n_items, config["k"]
        )
        seconds = time.perf_counter() - tick + retrieval_seconds
        record_grid(output, "Hybrid", weights, metrics, seconds)
        if metrics[f"MAP@{config['k']}"] > best_score:
            best_score, best_recs, best_seconds = (
                metrics[f"MAP@{config['k']}"],
                recs,
                seconds,
            )
            selected = weights
    record_result(output, period, "Hybrid", best_recs, best_seconds, config["k"])
    return selected


def tune_ranker(training, validation, period, config, output, retrieval_seconds):
    best_score = -1.0
    for params in config["ranker"]:
        print(f"Fitting ranker {params}", flush=True)
        tick = time.perf_counter()
        ranker = CandidateRanker(**params, random_state=config["seed"]).fit(training)
        recs = ranker.recommend(validation, config["k"])
        metrics = ranking_metrics(
            recs, period.targets, period.data.n_items, config["k"]
        )
        seconds = time.perf_counter() - tick + retrieval_seconds
        record_grid(output, "CatBoost", params, metrics, seconds)
        if metrics[f"MAP@{config['k']}"] > best_score:
            best_score, best_recs, best_seconds = (
                metrics[f"MAP@{config['k']}"],
                recs,
                seconds,
            )
            selected = params
    record_result(output, period, "CatBoost", best_recs, best_seconds, config["k"])
    return selected


def evaluate_final_models(period, items, selected, config, output):
    candidate_models = {}
    for name in config["models"]:
        print(f"Fitting final {name}", flush=True)
        tick = time.perf_counter()
        model = fit_model(name, selected[name], period.data, items, config["seed"])
        recs = model.recommend(period.targets, config["k"])
        record_result(
            output, period, name, recs, time.perf_counter() - tick, config["k"]
        )
        if selected[name] == config["candidate_models"].get(name):
            candidate_models[name] = model
    return candidate_models


def evaluate_final_ranker(
    training, validation, test, period, selected, config, output, seconds
):
    # A user in two time windows forms two separate ranking queries.
    training = training.assign(query_id=training.user_id.astype("int64") * 2)
    validation = validation.assign(query_id=validation.user_id.astype("int64") * 2 + 1)
    combined = pd.concat([training, validation], ignore_index=True)
    print("Refitting selected ranker on two past windows", flush=True)
    tick = time.perf_counter()
    ranker = CandidateRanker(**selected["CatBoost"], random_state=config["seed"]).fit(
        combined
    )
    recs = ranker.recommend(test, config["k"])
    record_result(
        output,
        period,
        "CatBoost",
        recs,
        time.perf_counter() - tick + seconds,
        config["k"],
    )
    pd.DataFrame(
        {
            "feature": ranker.features,
            "importance": ranker.model.get_feature_importance(
                type="PredictionValuesChange"
            ),
        }
    ).sort_values("importance", ascending=False).to_csv(
        output / "feature_importance.csv", index=False
    )
    return recs, {
        "ranker_training_rows": ranker.training_rows,
        "ranker_training_queries": int(ranker.training_queries),
    }


def save_example(recs, period, items, output):
    activity = period.positive.groupby("user_id")["item_id"].nunique()
    eligible = set(activity.index[activity.ge(5)]) & set(period.targets)
    user = min(eligible or set(period.targets))
    items.set_index("item_id").loc[
        recs[user], ["title", "genres"]
    ].reset_index().to_csv(output / "example_recommendations.csv", index=False)
    period.positive.loc[period.positive.user_id.eq(user)].merge(
        items[["item_id", "title"]], on="item_id"
    ).to_csv(output / "example_history.csv", index=False)
    return user


def input_manifest(data_dir, interactions):
    hashes = {}
    for name in ["interactions.csv", "items.csv", "users.csv"]:
        with (data_dir / name).open("rb") as source:
            hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "implicit",
                "catboost",
            ]
        },
        "input_sha256": hashes,
        "raw_rows": len(interactions),
        "raw_users": interactions.user_id.nunique(),
        "raw_items": interactions.item_id.nunique(),
    }


def run_experiment(data_dir: Path, config_path: Path, output: Path) -> None:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    for filename in [
        "metrics.csv",
        "segments.csv",
        "validation_grid.csv",
        "candidate_recall.csv",
        "run.json",
    ]:
        (output / filename).unlink(missing_ok=True)
    config = json.loads(config_path.read_text())
    save_json(config, output / "config.json")
    interactions = load_interactions(data_dir)
    items = pd.read_csv(data_dir / "items.csv")
    manifest = input_manifest(data_dir, interactions)
    threshold = config["min_watched_pct"]
    validation = make_period(
        interactions, VALIDATION_START, TEST_START, "validation", threshold
    )
    selected, models = tune_models(validation, items, config, output)
    validation_features, seconds = make_candidates(
        validation, items, config, output, models
    )
    del models
    selected["Hybrid"] = tune_hybrid(
        validation_features, validation, config, output, seconds
    )
    rank_train = make_period(
        interactions, RANKER_START, VALIDATION_START, "rank_train", threshold
    )
    training_features, _ = make_candidates(rank_train, items, config, output)
    selected["CatBoost"] = tune_ranker(
        training_features, validation_features, validation, config, output, seconds
    )
    save_json(selected, output / "selected_params.json")
    save_json(
        {
            "selection_metric": f"MAP@{config['k']}",
            "parameters": selected,
            "status": "frozen_before_test",
        },
        output / "selection.json",
    )
    manifest.update(
        validation_users=len(validation.targets),
        validation_catalog=validation.data.n_items,
    )
    del validation, rank_train
    test = make_period(interactions, TEST_START, TEST_END, "test", threshold)
    models = evaluate_final_models(test, items, selected, config, output)
    test_features, seconds = make_candidates(test, items, config, output, models)
    del models
    tick = time.perf_counter()
    hybrid = hybrid_recommendations(test_features, selected["Hybrid"], config["k"])
    record_result(
        output,
        test,
        "Hybrid",
        hybrid,
        time.perf_counter() - tick + seconds,
        config["k"],
    )
    recs, ranker_stats = evaluate_final_ranker(
        training_features,
        validation_features,
        test_features,
        test,
        selected,
        config,
        output,
        seconds,
    )
    user = save_example(recs, test, items, output)
    catalog = set(test.data.item_ids)
    manifest.update(
        **ranker_stats,
        test_users=len(test.targets),
        test_catalog=test.data.n_items,
        test_unseen_item_targets=sum(
            len(targets - catalog) for targets in test.targets.values()
        ),
        example_user=user,
        elapsed_seconds=time.perf_counter() - started,
        status="completed",
        sampling="none",
    )
    save_json(manifest, output / "run.json")
    print(f"Completed in {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/full"))
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run_experiment(args.data_dir, args.config, args.output)


if __name__ == "__main__":
    main()
