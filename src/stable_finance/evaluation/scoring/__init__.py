"""Stage 5, how good each stage's output is: the IC of a forecast, the P&L and Sharpe of a portfolio, how much of the forecast reached the weights (the transfer coefficient), execution-quality curves and implementation shortfall, and the statistics that keep a sweep of many configurations honest."""

from stable_finance.evaluation.scoring.exposure import PortfolioExposure, portfolio_exposure
from stable_finance.evaluation.scoring.ic import (
    InformationCoefficient,
    evaluate_forward_returns,
    grouped_rank_ic,
    grouped_rank_ic_by_label,
    paired_difference,
    pooled_estimates,
)
from stable_finance.evaluation.scoring.pnl import (
    BacktestResult,
    evaluate_fills,
    evaluate_weights,
    positions_from_fills,
)
from stable_finance.evaluation.scoring.sharpe import (
    ExecutionQualityCurve,
    breakeven_efq,
    cross_spread_sharpe,
    mid_price_sharpe,
    sharpe_ratio,
    sharpe_vs_execution_quality,
)
from stable_finance.evaluation.scoring.shortfall import (
    ImplementationShortfall,
    implementation_shortfall,
)
from stable_finance.evaluation.scoring.stats import (
    PBOResult,
    SharpeDifference,
    deflated_sharpe,
    deflated_sharpe_from_trials,
    expected_max_sharpe,
    min_track_record,
    pbo,
    probabilistic_sharpe,
    sharpe_difference,
)
from stable_finance.evaluation.scoring.tc import (
    TransferCoefficient,
    transfer_coefficient,
)

__all__ = [
    "BacktestResult",
    "ExecutionQualityCurve",
    "ImplementationShortfall",
    "InformationCoefficient",
    "PBOResult",
    "PortfolioExposure",
    "SharpeDifference",
    "TransferCoefficient",
    "breakeven_efq",
    "cross_spread_sharpe",
    "deflated_sharpe",
    "deflated_sharpe_from_trials",
    "evaluate_fills",
    "evaluate_forward_returns",
    "evaluate_weights",
    "expected_max_sharpe",
    "grouped_rank_ic",
    "grouped_rank_ic_by_label",
    "implementation_shortfall",
    "mid_price_sharpe",
    "min_track_record",
    "paired_difference",
    "pbo",
    "pooled_estimates",
    "portfolio_exposure",
    "positions_from_fills",
    "probabilistic_sharpe",
    "sharpe_difference",
    "sharpe_ratio",
    "sharpe_vs_execution_quality",
    "transfer_coefficient",
]
