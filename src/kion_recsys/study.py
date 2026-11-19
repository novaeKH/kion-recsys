"""Recent popularity, feature ablation and three chronological evaluation folds."""

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from kion_recsys.data import load_interactions
from kion_recsys.experiment import (
    append_rows,
    fit_model,
    input_manifest,
    make_period,
    save_json,
)
from kion_recsys.metrics import candidate_recall, ranking_metrics
from kion_recsys.models import PopularRecommender, RecentPopularRecommender
from kion_recsys.ranking import (
    CandidateRanker,
    add_labels,
    generate_candidates,
    ranking_features,
)

WEEK = pd.Timedelta(days=7)


def build_snapshots(interactions, items, cutoffs, config, output, cache_dir):
    summary = []
    for cutoff in cutoffs:
        print(f"Building snapshot {cutoff.date()}", flush=True)
        period = make_period(
            interactions,
            cutoff,
            cutoff + WEEK,
            str(cutoff.date()),
            config["min_watched_pct"],
        )
        models = {
            name: fit_model(name, params, period.data, items, config["seed"])
            for name, params in config["candidate_models"].items()
        }
        candidates = generate_candidates(
            models, period.targets, config["candidate_depth"]
        )
        features = ranking_features(
            candidates, period.positive, items, cutoff, models["ALS"]
        )
        features.to_pickle(cache_dir / f"{cutoff.date()}.pkl")
        summary.append(
            {
                "cutoff": cutoff.date(),
                "users": len(period.targets),
                "catalog": period.data.n_items,
                "candidate_rows": len(features),
                "candidate_recall": candidate_recall(candidates, period.targets),
            }
        )
        pd.DataFrame(summary).to_csv(output / "snapshots.csv", index=False)
        del models, candidates, features, period
        gc.collect()


def load_window(interactions, cutoff, config, cache_dir):
    period = make_period(
        interactions,
        cutoff,
        cutoff + WEEK,
        str(cutoff.date()),
        config["min_watched_pct"],
    )
    return period, pd.read_pickle(cache_dir / f"{cutoff.date()}.pkl")


def evaluate_popularity(period, windows, k):
    models = {"Popularity": PopularRecommender().fit(period.data)}
    models.update(
        {
            f"RecentPopularity_{days}": RecentPopularRecommender(days).fit(
                period.data, period.positive, period.cutoff
            )
            for days in windows
        }
    )
    rows = []
    for name, model in models.items():
        recs = model.recommend(period.targets, k)
        metrics = ranking_metrics(recs, period.targets, period.data.n_items, k)
        rows.append({"model": name, **metrics})
    return rows


def tune_ranker(train, validation, period, config, feature_groups, ablate):
    rows, best_params, best_score = [], None, -1.0
    for params in config["ranker"]:
        model = CandidateRanker(**params, random_state=config["seed"]).fit(
            train, feature_groups["full"]
        )
        recs = model.recommend(validation, config["k"])
        metrics = ranking_metrics(
            recs, period.targets, period.data.n_items, config["k"]
        )
        rows.append({"model": "CatBoost_full", "params": json.dumps(params), **metrics})
        if metrics[f"MAP@{config['k']}"] > best_score:
            best_params, best_score = params, metrics[f"MAP@{config['k']}"]
    if ablate:
        for group in ["activity", "scores"]:
            model = CandidateRanker(**best_params, random_state=config["seed"]).fit(
                train, feature_groups[group]
            )
            recs = model.recommend(validation, config["k"])
            metrics = ranking_metrics(
                recs, period.targets, period.data.n_items, config["k"]
            )
            rows.append(
                {
                    "model": f"CatBoost_{group}",
                    "params": json.dumps(best_params),
                    **metrics,
                }
            )
    return best_params, rows


def evaluate_rankers(train, validation, test, period, params, config, groups):
    train = train.assign(query_id=train.user_id.astype("int64") * 2)
    validation = validation.assign(query_id=validation.user_id.astype("int64") * 2 + 1)
    combined = pd.concat([train, validation], ignore_index=True)
    rows = []
    for group, features in groups.items():
        print(f"Evaluating {period.cutoff.date()} CatBoost_{group}", flush=True)
        tick = time.perf_counter()
        model = CandidateRanker(**params, random_state=config["seed"]).fit(
            combined, features
        )
        recs = model.recommend(test, config["k"])
        metrics = ranking_metrics(
            recs, period.targets, period.data.n_items, config["k"]
        )
        rows.append(
            {
                "model": f"CatBoost_{group}",
                **metrics,
                "fit_predict_seconds": time.perf_counter() - tick,
            }
        )
        score = metrics[f"MAP@{config['k']}"]
        print(f"MAP@{config['k']} = {score:.5f}", flush=True)
    return rows


def run_fold(interactions, cutoff, config, study, output, cache_dir):
    ablate = str(cutoff.date()) == study["ablation_cutoff"]
    train_period, train = load_window(
        interactions, cutoff - 2 * WEEK, config, cache_dir
    )
    validation_period, validation = load_window(
        interactions, cutoff - WEEK, config, cache_dir
    )
    train = add_labels(train, train_period.targets)
    validation = add_labels(validation, validation_period.targets)
    popularity_rows = evaluate_popularity(
        validation_period, study["popularity_windows"], config["k"]
    )
    recent_rows = [row for row in popularity_rows if row["model"].startswith("Recent")]
    best_recent = max(recent_rows, key=lambda row: row[f"MAP@{config['k']}"])
    window = int(best_recent["model"].rsplit("_", 1)[1])
    params, ranker_rows = tune_ranker(
        train, validation, validation_period, config, study["feature_groups"], ablate
    )
    validation_rows = [
        {
            "test_start": str(cutoff.date()),
            "validation_start": str((cutoff - WEEK).date()),
            **row,
        }
        for row in popularity_rows + ranker_rows
    ]
    append_rows(pd.DataFrame(validation_rows), output / "validation.csv")
    selection = {
        "test_start": str(cutoff.date()),
        "validation_start": str((cutoff - WEEK).date()),
        "recent_window_days": window,
        "ranker_params": params,
    }
    # Each fold's choices are written before evaluating its next week.
    save_json(selection, output / f"selection_{cutoff.date()}.json")
    del train_period, validation_period
    gc.collect()
    test_period, test = load_window(interactions, cutoff, config, cache_dir)
    rows = evaluate_popularity(test_period, [window], config["k"])
    groups = (
        study["feature_groups"] if ablate else {"full": study["feature_groups"]["full"]}
    )
    rows.extend(
        evaluate_rankers(train, validation, test, test_period, params, config, groups)
    )
    rows = [
        {
            "test_start": str(cutoff.date()),
            "test_end": str((cutoff + WEEK).date()),
            "recent_window_days": window,
            **row,
        }
        for row in rows
    ]
    append_rows(pd.DataFrame(rows), output / "metrics.csv")


def run_study(data_dir, config_path, study_path, output, cache_dir):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["validation.csv", "metrics.csv", "run.json"]:
        (output / filename).unlink(missing_ok=True)
    config = json.loads(config_path.read_text())
    study = json.loads(study_path.read_text())
    save_json({"experiment": config, "study": study}, output / "config.json")
    interactions = load_interactions(data_dir)
    items = pd.read_csv(data_dir / "items.csv")
    manifest = input_manifest(data_dir, interactions)
    test_starts = [pd.Timestamp(date) for date in study["test_starts"]]
    cutoffs = sorted(
        {date - shift * WEEK for date in test_starts for shift in [0, 1, 2]}
    )
    build_snapshots(interactions, items, cutoffs, config, output, cache_dir)
    for cutoff in test_starts:
        print(f"Running fold {cutoff.date()}", flush=True)
        run_fold(interactions, cutoff, config, study, output, cache_dir)
        gc.collect()
    manifest.update(
        status="completed",
        sampling="none",
        folds=len(test_starts),
        elapsed_seconds=time.perf_counter() - started,
    )
    save_json(manifest, output / "run.json")
    print(f"Completed study in {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--study-config", type=Path, default=Path("configs/study.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/study"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/study"))
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        run_study(
            args.data_dir, args.config, args.study_config, args.output, args.cache_dir
        )


if __name__ == "__main__":
    main()
