"""Train and validate the Approach B neural baseline on prepared card duels.

By default this script loads and uses ONLY train and valid splits so the test
set never influences hyperparameter or epoch choices. Pass --evaluate-test
explicitly for the single final evaluation once the configuration is locked.
Supports single-seed (--seed) or multi-seed (--seeds 0 1 2) runs with mean/std
aggregation and 9-suit-pair diagnostic accuracy on validation/test.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from card_suit_duel.data.filter.filtering import load_validate_index
from card_suit_duel.data.mapping.definitions import DatasetSplit
from card_suit_duel.data.pairs.pair_loader import build_pair_loader
from card_suit_duel.data.pairs.pairs_reader import load_pairs
from card_suit_duel.data.preprocessing import build_image_transform
from card_suit_duel.data.serialization import write_json
from card_suit_duel.exceptions import CardSuitDuelError
from card_suit_duel.logging_config import LoggingManager
from card_suit_duel.models import (
    BaselineDuelNet,
    CachedCardPairDataset,
    CardTensorCache,
    aggregate_metric_runs,
    build_pair_suit_lookup,
    collect_hardware_info,
    evaluate_baseline,
    fit_baseline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREP_DIR = (
    PROJECT_ROOT
    / "results"
    / "pipeline_output"
    / "preparation"
    / "20261116T234548Z_35065d09"
)
DEFAULT_IMAGE_INDEX = DEFAULT_PREP_DIR / "03_deduplication" / "deduplicated_images.csv"
DEFAULT_PAIRS_DIR = DEFAULT_PREP_DIR / "04_pairs" / "20261116T234604Z_cb0f352f"
DEFAULT_SUBSETS_DIR = DEFAULT_PREP_DIR / "05_subsets" / "20261116T234605Z_31f635c7"

logger = LoggingManager.get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--percentage", type=int, choices=(20, 50, 100), default=20)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=120)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of seeds for multi-seed runs with mean/std summary (overrides --seed).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        default=False,
        help="Evaluate on the held-out test split once after training (default: off).",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--image-index", type=Path, default=DEFAULT_IMAGE_INDEX)
    parser.add_argument("--subsets-dir", type=Path, default=DEFAULT_SUBSETS_DIR)
    parser.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "baseline",
    )
    return parser.parse_args(argv)


def _run_single_seed(
    *,
    seed: int,
    args: argparse.Namespace,
    train_ds: CachedCardPairDataset,
    valid_ds: CachedCardPairDataset,
    valid_suit_lookup: dict[str, tuple[str, str]],
    test_ds: CachedCardPairDataset | None,
    test_suit_lookup: dict[str, tuple[str, str]] | None,
    output_dir: Path,
    run_id: str,
    image_size: tuple[int, int] | None,
    hardware: dict[str, Any],
    cache_seconds: float,
    cache_details: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = build_pair_loader(train_ds, batch_size=args.batch_size, seed=seed)
    valid_loader = build_pair_loader(valid_ds, batch_size=args.batch_size, seed=seed)

    model = BaselineDuelNet(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    fit_result = fit_baseline(
        model,
        train_loader,
        valid_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_valid_batches=args.max_eval_batches,
        valid_pair_suit_lookup=valid_suit_lookup,
    )

    test_metrics = None
    if args.evaluate_test and test_ds is not None:
        test_loader = build_pair_loader(test_ds, batch_size=args.batch_size, seed=seed)
        test_metrics = evaluate_baseline(
            model,
            test_loader,
            device=args.device,
            max_batches=args.max_eval_batches,
            pair_suit_lookup=test_suit_lookup,
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    if fit_result["best_state_dict"] is not None:
        torch.save(fit_result["best_state_dict"], output_dir / "best_model.pt")

    report = {
        "approach": "B_neural_baseline",
        "run_id": run_id,
        "training_percentage": args.percentage,
        "image_size": list(image_size) if image_size is not None else [224, 224],
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": seed,
        "epochs": args.epochs,
        "epochs_completed": fit_result["epochs_completed"],
        "patience": args.patience,
        "stopped_early": fit_result["stopped_early"],
        "hardware": hardware,
        "cache_seconds": cache_seconds,
        "cache_details": cache_details,
        "best_epoch": fit_result["best_epoch"],
        "best_valid_macro_f1": fit_result["best_valid_macro_f1"],
        "best_valid_loss": fit_result["best_valid_loss"],
        "best_valid_accuracy": fit_result["best_valid_metrics"]["accuracy"],
        "best_valid_ms_per_pair": fit_result["best_valid_metrics"]["ms_per_pair"],
        "best_valid_per_suit_combination": fit_result["best_valid_metrics"].get(
            "per_suit_combination"
        ),
        "total_training_seconds": fit_result["total_training_seconds"],
        "trainable_parameters": fit_result["trainable_parameters"],
        "history": fit_result["history"],
        "test_evaluated": bool(args.evaluate_test),
        "test": test_metrics,
    }
    write_json(report=report, output_path=output_dir / "summary.json")
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    LoggingManager.configure(levels=["INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"])

    try:
        seeds = list(args.seeds) if args.seeds is not None else [args.seed]
        hardware = collect_hardware_info(args.device)
        image_size = (args.image_size, args.image_size) if args.image_size > 0 else None
        transform = build_image_transform(image_size=image_size)

        logger.info("Loading deduplicated card index from %s", args.image_index.name)
        cards = load_validate_index(args.image_index)

        train_csv = args.subsets_dir / str(args.percentage) / "train_pairs.csv"
        valid_csv = args.pairs_dir / "valid_pairs.csv"

        train_pairs = load_pairs(train_csv, cards, expected_split=DatasetSplit.TRAIN)
        valid_pairs = load_pairs(valid_csv, cards, expected_split=DatasetSplit.VALID)
        valid_suit_lookup = build_pair_suit_lookup(valid_pairs)

        tensor_cache = CardTensorCache(args.dataset_root, transform=transform)
        train_ds = CachedCardPairDataset(train_pairs, args.dataset_root, tensor_cache=tensor_cache)
        valid_ds = CachedCardPairDataset(valid_pairs, args.dataset_root, tensor_cache=tensor_cache)

        # Pre-warm train and valid caches ONCE BEFORE training so epoch timings exclude disk I/O.
        logger.info("Pre-warming image tensor cache for train and valid splits")
        train_cache_seconds = train_ds.prewarm_cache()
        valid_cache_seconds = valid_ds.prewarm_cache()
        cache_seconds = train_cache_seconds + valid_cache_seconds

        test_ds = None
        test_suit_lookup = None
        test_cache_seconds = None
        if args.evaluate_test:
            logger.info("Pre-warming image tensor cache for held-out test split")
            test_csv = args.pairs_dir / "test_pairs.csv"
            test_pairs = load_pairs(test_csv, cards, expected_split=DatasetSplit.TEST)
            test_suit_lookup = build_pair_suit_lookup(test_pairs)
            test_ds = CachedCardPairDataset(test_pairs, args.dataset_root, tensor_cache=tensor_cache)
            test_cache_seconds = test_ds.prewarm_cache()

        cache_details = {
            "train_seconds": train_cache_seconds,
            "valid_seconds": valid_cache_seconds,
            "test_seconds": test_cache_seconds,
            "cached_unique_images": len(tensor_cache),
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_p{args.percentage}_{uuid4().hex[:8]}"
        root_output_dir = args.output_root / run_id

        logger.info(
            "Starting Baseline training (%d%% train = %d pairs, image_size=%s, epochs=%d, patience=%s, seeds=%s, device=%s, cache=%.2fs)",
            args.percentage,
            len(train_ds),
            image_size,
            args.epochs,
            args.patience,
            seeds,
            hardware["device"],
            cache_seconds,
        )

        if len(seeds) == 1:
            report = _run_single_seed(
                seed=seeds[0],
                args=args,
                train_ds=train_ds,
                valid_ds=valid_ds,
                valid_suit_lookup=valid_suit_lookup,
                test_ds=test_ds,
                test_suit_lookup=test_suit_lookup,
                output_dir=root_output_dir,
                run_id=run_id,
                image_size=image_size,
                hardware=hardware,
                cache_seconds=cache_seconds,
                cache_details=cache_details,
            )
            if report["test"] is not None:
                logger.success(
                    "Baseline completed | Best Valid Macro-F1: %.4f (epoch %d/%d) | Test Acc: %.4f | Test Macro-F1: %.4f | Cache: %.2fs | Train time: %.2fs",
                    report["best_valid_macro_f1"],
                    report["best_epoch"],
                    report["epochs_completed"],
                    report["test"]["accuracy"],
                    report["test"]["macro_f1"],
                    cache_seconds,
                    report["total_training_seconds"],
                )
            else:
                logger.success(
                    "Baseline completed (validation only; test not evaluated) | Best Valid Macro-F1: %.4f (epoch %d/%d) | Cache: %.2fs | Train time: %.2fs",
                    report["best_valid_macro_f1"],
                    report["best_epoch"],
                    report["epochs_completed"],
                    cache_seconds,
                    report["total_training_seconds"],
                )
            return 0

        root_output_dir.mkdir(parents=True, exist_ok=False)
        seed_reports: list[dict[str, Any]] = []
        for seed in seeds:
            logger.info("--- Running seed %d ---", seed)
            seed_report = _run_single_seed(
                seed=seed,
                args=args,
                train_ds=train_ds,
                valid_ds=valid_ds,
                valid_suit_lookup=valid_suit_lookup,
                test_ds=test_ds,
                test_suit_lookup=test_suit_lookup,
                output_dir=root_output_dir / f"seed_{seed}",
                run_id=f"{run_id}_seed{seed}",
                image_size=image_size,
                hardware=hardware,
                cache_seconds=cache_seconds,
                cache_details=cache_details,
            )
            seed_reports.append(seed_report)
            logger.info(
                "Seed %d | Best Valid Macro-F1: %.4f (epoch %d/%d) | Train time: %.2fs",
                seed,
                seed_report["best_valid_macro_f1"],
                seed_report["best_epoch"],
                seed_report["epochs_completed"],
                seed_report["total_training_seconds"],
            )

        aggregated: dict[str, Any] = {
            "approach": "B_neural_baseline_multiseed",
            "run_id": run_id,
            "training_percentage": args.percentage,
            "image_size": list(image_size) if image_size is not None else [224, 224],
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "seeds": seeds,
            "hardware": hardware,
            "cache_seconds": cache_seconds,
            "cache_details": cache_details,
            "trainable_parameters": seed_reports[0]["trainable_parameters"],
            "test_evaluated": bool(args.evaluate_test),
            "summary": {
                "valid_macro_f1": aggregate_metric_runs(
                    [r["best_valid_macro_f1"] for r in seed_reports]
                ),
                "valid_accuracy": aggregate_metric_runs(
                    [r["best_valid_accuracy"] for r in seed_reports]
                ),
                "best_epoch": aggregate_metric_runs(
                    [r["best_epoch"] for r in seed_reports]
                ),
                "total_training_seconds": aggregate_metric_runs(
                    [r["total_training_seconds"] for r in seed_reports]
                ),
                "valid_ms_per_pair": aggregate_metric_runs(
                    [r["best_valid_ms_per_pair"] for r in seed_reports]
                ),
            },
            "per_seed": [
                {
                    "seed": r["seed"],
                    "best_epoch": r["best_epoch"],
                    "epochs_completed": r["epochs_completed"],
                    "stopped_early": r["stopped_early"],
                    "best_valid_macro_f1": r["best_valid_macro_f1"],
                    "best_valid_accuracy": r["best_valid_accuracy"],
                    "best_valid_per_suit_combination": r["best_valid_per_suit_combination"],
                    "total_training_seconds": r["total_training_seconds"],
                    "test_accuracy": r["test"]["accuracy"] if r["test"] is not None else None,
                    "test_macro_f1": r["test"]["macro_f1"] if r["test"] is not None else None,
                }
                for r in seed_reports
            ],
        }
        if args.evaluate_test:
            aggregated["summary"]["test_macro_f1"] = aggregate_metric_runs(
                [r["test"]["macro_f1"] for r in seed_reports]
            )
            aggregated["summary"]["test_accuracy"] = aggregate_metric_runs(
                [r["test"]["accuracy"] for r in seed_reports]
            )
            aggregated["summary"]["test_ms_per_pair"] = aggregate_metric_runs(
                [r["test"]["ms_per_pair"] for r in seed_reports]
            )

        write_json(report=aggregated, output_path=root_output_dir / "summary.json")
        v_f1 = aggregated["summary"]["valid_macro_f1"]
        t_sec = aggregated["summary"]["total_training_seconds"]
        logger.success(
            "Multi-seed Baseline completed (%d seeds) | Valid Macro-F1: %.4f ± %.4f | Avg Train time: %.2fs | Cache: %.2fs",
            len(seeds),
            v_f1["mean"],
            v_f1["std"],
            t_sec["mean"],
            cache_seconds,
        )
        return 0
    except (CardSuitDuelError, OSError, ValueError, RuntimeError) as exc:
        logger.error("Baseline training failed [%s]: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
