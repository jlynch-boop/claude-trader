"""
config.py — Load and validate config.yaml.

Usage:
    from src.config import load_config
    cfg = load_config()
    print(cfg['trading']['pairs'])
"""

import os
import yaml


def load_config(path: str = None) -> dict:
    """
    Load config.yaml from the project root.

    Args:
        path: Optional explicit path to config.yaml.
              Defaults to config.yaml in the project root.

    Returns:
        Parsed config as a nested dictionary.

    Raises:
        FileNotFoundError: If config.yaml doesn't exist.
        ValueError: If required keys are missing.
    """
    if path is None:
        # Walk up from this file's location to find project root
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        path = os.path.join(root, "config.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config.yaml not found at {path}. "
            "Make sure you're running from the project root."
        )

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    """Check that all required top-level sections exist."""
    required_sections = ["exchange", "trading", "backtest", "risk", "strategies"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(
                f"Missing required section '{section}' in config.yaml"
            )

    # Validate trading pairs list is not empty
    pairs = cfg.get("trading", {}).get("pairs", [])
    if not pairs:
        raise ValueError("config.yaml: trading.pairs must contain at least one pair")

    # Validate starting capital is positive
    capital = cfg.get("trading", {}).get("starting_capital", 0)
    if capital <= 0:
        raise ValueError(
            f"config.yaml: trading.starting_capital must be > 0, got {capital}"
        )

    # Validate risk percentages are between 0 and 1
    risk = cfg.get("risk", {})
    for key in ["max_position_pct", "max_drawdown_pct", "stop_loss_pct",
                "take_profit_pct", "risk_per_trade_pct"]:
        val = risk.get(key, 0)
        if not (0 < val < 1):
            raise ValueError(
                f"config.yaml: risk.{key} must be between 0 and 1, got {val}"
            )
