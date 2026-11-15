"""Configurable 100-lot TQQQ book, priced by this repo's Black-Scholes engine."""

from tqqq_book.desk import BookConfig, Tenor, run

__all__ = ["BookConfig", "Tenor", "run"]
