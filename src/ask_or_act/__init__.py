"""ask-or-act: evaluate and calibrate AI decisions, and decide when to act, ask or escalate.

The public API (docs/cli-design.md §5.3):

    import ask_or_act as aoa
    th = aoa.find_thresholds("preds.csv", alpha=0.1)

Names load on first use, so ``import ask_or_act`` (and every CLI command) stays fast.
"""

from importlib import import_module
from importlib.metadata import version
from typing import Any

__version__ = version("ask-or-act")

_PUBLIC = {
    "find_thresholds": "ask_or_act.api",
    "evaluate": "ask_or_act.api",
    "compare": "ask_or_act.api",
    "Thresholds": "ask_or_act.api",
    "Evaluation": "ask_or_act.api",
    "Comparison": "ask_or_act.api",
    "NotEnoughData": "ask_or_act.api",
    "read_predictions": "ask_or_act.predictions",
    "check": "ask_or_act.predictions",
    "CheckResult": "ask_or_act.predictions",
    "InputError": "ask_or_act.predictions",
    "Problem": "ask_or_act.predictions",
    "DecisionTable": "ask_or_act.report.table",
    "DecisionRow": "ask_or_act.report.table",
}

__all__ = ["__version__", *_PUBLIC]


def __getattr__(name: str) -> Any:
    module = _PUBLIC.get(name)
    if module is None:
        raise AttributeError(f"module 'ask_or_act' has no attribute {name!r}")
    return getattr(import_module(module), name)
