"""Train and validate the Approach A DeepProbLog model on prepared card duels.

Mirrors scripts/train_baseline.py: same data run, same subsets, same image
cache and preprocessing, same metrics and the same best-validation checkpoint
rule. By default only train and valid are loaded; pass --evaluate-test once
the configuration is locked for the single final evaluation.
"""

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from card_suit_duel.data.filter.filtering import load_validate_index
from card_suit_duel.data.mapping.definitions import DatasetSplit
from card_suit_duel.data.pairs.pairs_reader import load_pairs
from card_suit_duel.data.preprocessing import build_image_transform
from card_suit_duel.data.serialization import write_json
from card_suit_duel.exceptions import CardSuitDuelError
from card_suit_duel.logging_config import LoggingManager
from card_suit_duel.models import CardTensorCache, SuitNet, collect_hardware_info
from card_suit_duel.models.deepproblog_adapter import (
    DuelQueries,
    build_deepproblog_model,
    register_split,
)
from card_suit_duel.models.deepproblog_trainer import (
    evaluate_deepproblog,
    evaluate_suit_predictions,
    fit_deepproblog,
    load_checkpoint,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        default=False,
        help="Evaluate on the held-out test split once after training (default: off).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from a checkpoints/epoch_NNN.pt file; the run continues in its directory. "
        "Pass the same data and hyperparameter flags as the original run.",
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
        default=PROJECT_ROOT / "results" / "runs" / "deepproblog",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    LoggingManager.configure(levels=["INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"])

    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        hardware = collect_hardware_info("cpu")
        image_size = (args.image_size, args.image_size) if args.image_size > 0 else None
        transform = build_image_transform(image_size=image_size)

        logger.info("Loading deduplicated card index from %s", args.image_index.name)
        cards = load_validate_index(args.image_index)

        train_csv = args.subsets_dir / str(args.percentage) / "train_pairs.csv"
        valid_csv = args.pairs_dir / "valid_pairs.csv"
        train_pairs = load_pairs(train_csv, cards, expected_split=DatasetSplit.TRAIN)
        valid_pairs = load_pairs(valid_csv, cards, expected_split=DatasetSplit.VALID)

        train_ds = DuelQueries(train_pairs, DatasetSplit.TRAIN)
        valid_ds = DuelQueries(valid_pairs, DatasetSplit.VALID)

        model = build_deepproblog_model(SuitNet(), lr=args.lr)
        tensor_cache = CardTensorCache(args.dataset_root, transform=transform)

        # Pre-warm train and valid caches BEFORE training so epoch timings exclude disk I/O.
        logger.info("Pre-warming image tensor cache for train and valid splits")
        train_cache_seconds = register_split(model, train_ds, tensor_cache)
        valid_cache_seconds = register_split(model, valid_ds, tensor_cache)
        cache_seconds = train_cache_seconds + valid_cache_seconds

        resume_state = None
        if args.resume_from is not None:
            resume_state = load_checkpoint(args.resume_from)
            output_dir = args.resume_from.resolve().parent.parent
            run_id = output_dir.name
            logger.info("Resuming run %s after epoch %d", run_id, resume_state["epoch"])
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"{timestamp}_p{args.percentage}_{uuid4().hex[:8]}"
            output_dir = args.output_root / run_id
            output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_dir = output_dir / "checkpoints"
        logger.info("Writing per-epoch checkpoints to %s", checkpoint_dir)

        logger.info(
            "Starting DeepProbLog training (%d%% train = %d pairs, image_size=%s, epochs=%d, cache=%.2fs)",
            args.percentage,
            len(train_ds),
            image_size,
            args.epochs,
            cache_seconds,
        )
        fit_result = fit_deepproblog(
            model,
            train_ds,
            valid_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            max_train_batches=args.max_train_batches,
            max_valid_batches=args.max_eval_batches,
            checkpoint_dir=checkpoint_dir,
            resume_state=resume_state,
        )

        # Diagnostic only: suit labels are never used for training or model selection.
        valid_suits = evaluate_suit_predictions(model, valid_ds.cards, tensor_cache)

        test_metrics = None
        test_suits = None
        test_cache_seconds = None
        if args.evaluate_test:
            logger.info("Evaluating best validation checkpoint on held-out test split")
            test_csv = args.pairs_dir / "test_pairs.csv"
            test_pairs = load_pairs(test_csv, cards, expected_split=DatasetSplit.TEST)
            test_ds = DuelQueries(test_pairs, DatasetSplit.TEST)
            test_cache_seconds = register_split(model, test_ds, tensor_cache)
            test_metrics = evaluate_deepproblog(model, test_ds, max_batches=args.max_eval_batches)
            test_suits = evaluate_suit_predictions(model, test_ds.cards, tensor_cache)

        if fit_result["best_state_dict"] is not None:
            torch.save(fit_result["best_state_dict"], output_dir / "best_model.pt")

        report = {
            "approach": "A_deepproblog",
            "run_id": run_id,
            "training_percentage": args.percentage,
            "image_size": list(image_size) if image_size is not None else [224, 224],
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "epochs": args.epochs,
            "engine": "ExactEngine",
            "checkpoint_dir": str(checkpoint_dir),
            "resumed_from": str(args.resume_from) if args.resume_from is not None else None,
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
            "valid_suit_diagnostics": valid_suits,
            "test_evaluated": bool(args.evaluate_test),
            "test": test_metrics,
            "test_suit_diagnostics": test_suits,
        }
        write_json(report=report, output_path=output_dir / "summary.json")

        summary = (
            "DeepProbLog completed | Best Valid Macro-F1: %.4f (epoch %d) | "
            "Valid suit acc (up to rotation): %.4f | Cache: %.2fs | Train time: %.2fs"
        )
        values = [
            fit_result["best_valid_macro_f1"],
            fit_result["best_epoch"],
            valid_suits["accuracy_up_to_rotation"],
            cache_seconds,
            fit_result["total_training_seconds"],
        ]
        if test_metrics is not None:
            summary += " | Test Acc: %.4f | Test Macro-F1: %.4f"
            values += [test_metrics["accuracy"], test_metrics["macro_f1"]]
        logger.success(summary, *values)
        return 0
    except (CardSuitDuelError, OSError, ValueError, RuntimeError) as exc:
        logger.error("DeepProbLog training failed [%s]: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
