"""Inspect this checkout using the installed card_suit_duel package."""

from pathlib import Path

from card_suit_duel.data.inspection import main


if __name__ == "__main__":
    raise SystemExit(main(project_root=Path(__file__).resolve().parent.parent))
