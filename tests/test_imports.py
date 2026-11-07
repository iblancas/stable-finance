"""The package's import order: a cycle shows only from the entry point that closes it."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import stable_finance

EVALUATION = Path(stable_finance.__file__).parent / "evaluation"

#: The modules on the way between the configuration, the backtest and the environment.
ENTRY_POINTS = ("config", "execution", "preparation", "backtest", "scoring.shortfall", "trades",
                "trades.rebalance", "trades.rl", "trades.rl.env", "trades.rl.replay")

#: The code that sits on the harness and must reach it through what it publishes.
PUBLIC_ONLY = ("preparation.py", "scoring/shortfall.py", "trades/rl/env.py", "trades/rl/features.py",
               "trades/rl/gym.py", "trades/rl/replay.py")


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_every_entry_point_imports_first(module):
    done = subprocess.run([sys.executable, "-c", f"import stable_finance.evaluation.{module}"],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("path", PUBLIC_ONLY)
def test_nothing_private_is_borrowed(path):
    tree = ast.parse((EVALUATION / path).read_text())
    borrowed = [alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("stable_finance")
                for alias in node.names if alias.name.startswith("_")]
    assert not borrowed
