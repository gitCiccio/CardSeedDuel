"""Train and validate the Approach B neural baseline on prepared card duels.

By default this script loads and uses ONLY train and valid splits so the test
set never influences hyperparameter or epoch choices. Pass --evaluate-test
explicitly for the single final evaluation once the configuration is locked.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=120)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    LoggingManager.configure(levels=["INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"])

    try:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        hardware = collect_hardware_info(args.device)
        image_size = (args.image_size, args.image_size) if args.image_size > 0 else None
        transform = build_image_transform(image_size=image_size)

        logger.info("Loading deduplicated card index from %s", args.image_index.name)
        cards = load_validate_index(args.image_index)

        train_csv = args.subsets_dir / str(args.percentage) / "train_pairs.csv"
        valid_csv = args.pairs_dir / "valid_pairs.csv"

        train_pairs = load_pairs(train_csv, cards, expected_split=DatasetSplit.TRAIN)
        valid_pairs = load_pairs(valid_csv, cards, expected_split=DatasetSplit.VALID)

        tensor_cache = CardTensorCache(args.dataset_root, transform=transform)
        train_ds = CachedCardPairDataset(train_pairs, args.dataset_root, tensor_cache=tensor_cache)
        valid_ds = CachedCardPairDataset(valid_pairs, args.dataset_root, tensor_cache=tensor_cache)

        # Pre-warm train and valid caches BEFORE training so epoch timings exclude disk I/O.
        logger.info("Pre-warming image tensor cache for train and valid splits")
        train_cache_seconds = train_ds.prewarm_cache()
        valid_cache_seconds = valid_ds.prewarm_cache()
        cache_seconds = train_cache_seconds + valid_cache_seconds

        train_loader = build_pair_loader(train_ds, batch_size=args.batch_size, seed=args.seed)
        valid_loader = build_pair_loader(valid_ds, batch_size=args.batch_size, seed=args.seed)

        model = BaselineDuelNet(
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )

        logger.info(
            "Starting Baseline training (%d%% train = %d pairs, image_size=%s, epochs=%d, device=%s, cache=%.2fs)",
            args.percentage,
            len(train_ds),
            image_size,
            args.epochs,
            hardware["device"],
            cache_seconds,
        )
        fit_result = fit_baseline(
            model,
            train_loader,
            valid_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            max_train_batches=args.max_train_batches,
            max_valid_batches=args.max_eval_batches,
        )

        test_metrics = None
        test_cache_seconds = None
        if args.evaluate_test:
            logger.info("Evaluating best validation checkpoint on held-out test split")
            test_csv = args.pairs_dir / "test_pairs.csv"
            test_pairs = load_pairs(test_csv, cards, expected_split=DatasetSplit.TEST)
            test_ds = CachedCardPairDataset(test_pairs, args.dataset_root, tensor_cache=tensor_cache)
            test_cache_seconds = test_ds.prewarm_cache()
            test_loader = build_pair_loader(test_ds, batch_size=args.batch_size, seed=args.seed)
            test_metrics = evaluate_baseline(
                model,
                test_loader,
                device=args.device,
                max_batches=args.max_eval_batches,
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_p{args.percentage}_{uuid4().hex[:8]}"
        output_dir = args.output_root / run_id
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
            "seed": args.seed,
            "epochs": args.epochs,
            "hardware": hardware,
            "cache_seconds": cache_seconds,
            "cache_details": {
                "train_seconds": train_cache_seconds,
                "valid_seconds": valid_cache_seconds,
                "test_seconds": test_cache_seconds,
                "cached_unique_images": len(tensor_cache),
            },
            "best_epoch": fit_result["best_epoch"],
            "best_valid_macro_f1": fit_result["best_valid_macro_f1"],
            "best_valid_loss": fit_result["best_valid_loss"],
            "total_training_seconds": fit_result["total_training_seconds"],
            "trainable_parameters": fit_result["trainable_parameters"],
            "history": fit_result["history"],
            "test_evaluated": bool(args.evaluate_test),
            "test": test_metrics,
        }
        write_json(report=report, output_path=output_dir / "summary.json")

        if test_metrics is not None:
            logger.success(
                "Baseline completed | Best Valid Macro-F1: %.4f (epoch %d) | Test Acc: %.4f | Test Macro-F1: %.4f | Cache: %.2fs | Train time: %.2fs",
                fit_result["best_valid_macro_f1"],
                fit_result["best_epoch"],
                test_metrics["accuracy"],
                test_metrics["macro_f1"],
                cache_seconds,
                fit_result["total_training_seconds"],
            )
        else:
            logger.success(
                "Baseline completed (validation only; test not evaluated) | Best Valid Macro-F1: %.4f (epoch %d) | Cache: %.2fs | Train time: %.2fs",
                fit_result["best_valid_macro_f1"],
                fit_result["best_epoch"],
                cache_seconds,
                fit_result["total_training_seconds"],
            )
        return 0
    except (CardSuitDuelError, OSError, ValueError, RuntimeError) as exc:
        logger.error("Baseline training failed [%s]: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
