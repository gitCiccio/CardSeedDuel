"""Rebuild the dataset indices from one versioned configuration."""
import argparse
from pathlib import Path
from card_suit_duel.data.preparation import run_preparation
from card_suit_duel.exceptions import DatasetError
from card_suit_duel.logging_config import LoggingManager

logger = LoggingManager.get_logger(__name__)


def main():
    """Run preparation; preserve raw data and return a concise failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/preparation.json"))
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else args.project_root / args.config
    LoggingManager.configure(levels=["SUCCESS", "WARNING", "ERROR"])
    try:
        output = run_preparation(args.project_root, config)
        LoggingManager.configure(level="INFO", log_file=output / "execution.log")
        logger.success("Dataset preparation completed: %s", output.name)
        logger.info("Reproducibility manifest: reproducibility.json")
        return 0
    except (DatasetError, OSError, ValueError, RuntimeError) as exc:
        logger.error("Preparation failed [%s]: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
