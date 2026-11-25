"""Symbolic game rules (ProbLog) for the DeepProbLog approach."""

from pathlib import Path

PROGRAM_PATH = Path(__file__).resolve().parent / "card_duel.pl"
SUIT_NETWORK_NAME = "suit_net"
DUEL_PREDICATE = "duel"

__all__ = ["DUEL_PREDICATE", "PROGRAM_PATH", "SUIT_NETWORK_NAME"]
