"""Algothon 2026 — Version 1.

Adaptive one-day momentum/reversal strategy:
- Estimate whether one-day momentum has recently worked over the latest
  100 completed signal/outcome pairs.
- Use momentum when that trailing estimate is positive and reversal when
  it is negative.
- Hold equal dollar exposure per instrument, with the larger permitted
  allocation for instrument 0.

The function is deliberately stateless: every decision is computed only from
``prcSoFar`` supplied by the evaluator.
"""

import numpy as np


REGIME_WINDOW = 100
DOLLARS_PER_INSTRUMENT = 5_000.0
DOLLARS_INSTRUMENT_0 = 50_000.0


def _trailing_regime_score(prices: np.ndarray) -> float:
    """Return the recent average payoff of one-day momentum.

    If ``r_t`` is the return observed at day t, a one-day momentum signal
    formed then earns ``sign(r_t) * r_{t+1}``. Only completed pairs are used,
    so this calculation does not use future information.
    """
    returns = np.log(prices[:, 1:] / prices[:, :-1])
    completed_payoffs = np.sign(returns[:, :-1]) * returns[:, 1:]
    recent_payoffs = completed_payoffs[:, -REGIME_WINDOW:]
    return float(np.mean(recent_payoffs))


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """Return target share positions for all instruments."""
    n_instruments, n_days = prcSoFar.shape

    # Three prices are required to form at least one completed
    # signal/outcome pair. Stay flat during the warm-up period.
    if n_days < 3:
        return np.zeros(n_instruments, dtype=int)

    current_prices = prcSoFar[:, -1]

    # Defensive handling for malformed prices, although the supplied data is
    # expected to contain strictly positive finite values.
    if np.any(~np.isfinite(current_prices)) or np.any(current_prices <= 0):
        return np.zeros(n_instruments, dtype=int)

    regime_score = _trailing_regime_score(prcSoFar)
    regime = np.sign(regime_score)

    last_return = np.log(current_prices / prcSoFar[:, -2])
    signal = regime * np.sign(last_return)

    target_dollars = np.full(n_instruments, DOLLARS_PER_INSTRUMENT)
    target_dollars[0] = DOLLARS_INSTRUMENT_0

    target_shares = signal * target_dollars / current_prices
    return np.trunc(target_shares).astype(int)
