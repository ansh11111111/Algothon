import numpy as np

nInst = 51

# ---- Core pairs strategy parameters (tuned via grid search in sweep.py / strategies.py) ----
Z_LOOKBACK = 40
PAIR_DOLLAR_BUDGET = 8000
MIN_HIST = Z_LOOKBACK + 2

# ---- RSI overlay parameters ----
# Applying RSI to ALL 51 instruments lifted Sharpe 3.57 -> 3.65, but was inconsistent
# (went slightly negative at a 300-day test window) because it fades names that actually
# exhibit momentum, not mean reversion, in their own price history.
# Restricting the overlay to the 15 instruments with genuinely negative lag-1 return
# autocorrelation (classified using only the first ~250 days, to avoid lookahead bias)
# is both stronger (3.57 -> 3.87) AND more robust: positive across every test window
# checked (100/150/200/250/300 days), unlike the unfiltered version.
RSI_LOOKBACK = 7
RSI_VOL_LOOKBACK = 10
OVERLAY_BUDGET = 55000

# Indices (into the 51-instrument price array) of names with train-period lag-1
# autocorrelation < -0.02: ANSO, MHRM, AGVF, HTRK, NGTE, ACAC, NWIG, CUBO, FARS,
# MDGI, MSDP, AETS, LSST, ULXY, HUXZ
RANGE_BOUND_IDX = [16, 49, 23, 44, 45, 27, 20, 14, 48, 22, 12, 39, 2, 40, 8]

PAIRS = [
    {'i': 40, 'j': 47, 'beta': -1.189094, 'alpha': 82.495174},  # ULXY-FCSG
    {'i': 10, 'j': 46, 'beta': 1.532483, 'alpha': -6.107257},   # SMAH-ILVX
    {'i': 29, 'j': 37, 'beta': -0.085821, 'alpha': 23.025273},  # GARI-EELT
    {'i': 25, 'j': 46, 'beta': 1.880501, 'alpha': 26.585463},   # CTGI-ILVX
    {'i': 31, 'j': 43, 'beta': 0.698554, 'alpha': 2.946142},    # ACIX-ITPA
    {'i': 1, 'j': 20, 'beta': 0.654651, 'alpha': 34.771618},    # AENO-NWIG
    {'i': 15, 'j': 25, 'beta': 0.563477, 'alpha': -3.874114},   # HRET-CTGI
    {'i': 18, 'j': 38, 'beta': -0.695756, 'alpha': 57.608914},  # RTTH-HRND
    {'i': 37, 'j': 46, 'beta': 1.719904, 'alpha': 32.278693},   # EELT-ILVX
    {'i': 10, 'j': 25, 'beta': 0.627871, 'alpha': -10.038905},  # SMAH-CTGI
    {'i': 14, 'j': 40, 'beta': 1.568719, 'alpha': 49.573165},   # CUBO-ULXY
    {'i': 33, 'j': 42, 'beta': 0.746095, 'alpha': -1.319150},   # MTNS-BENI
    {'i': 35, 'j': 42, 'beta': 0.927989, 'alpha': 9.033595},    # NAYO-BENI
    {'i': 25, 'j': 37, 'beta': 0.805521, 'alpha': 18.543617},   # CTGI-EELT
    {'i': 0, 'j': 3, 'beta': 0.262566, 'alpha': 68.909759},     # ALGO-SRNA
    {'i': 49, 'j': 50, 'beta': 0.255513, 'alpha': 9.175624},    # MHRM-EAFC
    {'i': 25, 'j': 29, 'beta': -5.313396, 'alpha': 173.974749}, # CTGI-GARI
    {'i': 13, 'j': 45, 'beta': 0.539309, 'alpha': 4.035365},    # EORC-NGTE
    {'i': 10, 'j': 13, 'beta': 1.562014, 'alpha': -6.646338},   # SMAH-EORC
    {'i': 40, 'j': 50, 'beta': -0.885025, 'alpha': 82.522792},  # ULXY-EAFC
]


def pairsDollarSignal(prcSoFar):
    """Raw per-instrument dollar exposure from the 20-pair stat-arb book (pre share-rounding)."""
    nins, nt = prcSoFar.shape
    dollar = np.zeros(nins)
    if nt < MIN_HIST:
        return dollar

    for p in PAIRS:
        i, j, beta, alpha = p['i'], p['j'], p['beta'], p['alpha']
        spread_hist = prcSoFar[i, :] - beta * prcSoFar[j, :] - alpha
        recent = spread_hist[-Z_LOOKBACK:]
        mu, sd = recent.mean(), recent.std()
        if sd < 1e-8:
            continue
        z = (spread_hist[-1] - mu) / sd
        signal = -z

        price_i = prcSoFar[i, -1]
        price_j = prcSoFar[j, -1]
        pos_i_shares = signal * PAIR_DOLLAR_BUDGET / price_i
        pos_j_shares = -beta * pos_i_shares

        dollar[i] += pos_i_shares * price_i
        dollar[j] += pos_j_shares * price_j

    return dollar


def rsiDollarSignal(prcSoFar):
    """Per-instrument RSI mean-reversion signal (fades overbought/oversold), inverse-vol
    weighted. Returns a raw (un-budgeted) dollar-like signal to be normalized by the caller."""
    nins, nt = prcSoFar.shape
    if nt < RSI_LOOKBACK + 2:
        return np.zeros(nins)

    deltas = np.diff(prcSoFar[:, -(RSI_LOOKBACK + 1):], axis=1)
    gains = np.clip(deltas, 0, None).mean(axis=1)
    losses = np.clip(-deltas, 0, None).mean(axis=1)
    losses = np.where(losses < 1e-8, 1e-8, losses)
    rs = gains / losses
    rsi = 100 - 100 / (1 + rs)

    rets = np.diff(np.log(prcSoFar), axis=1)
    vl = min(RSI_VOL_LOOKBACK, rets.shape[1])
    vol = np.maximum(rets[:, -vl:].std(axis=1), 1e-8)

    signal = -(rsi - 50) / vol   # fade: overbought (RSI>50) -> short, oversold -> long

    mask = np.zeros(nins)
    mask[RANGE_BOUND_IDX] = 1.0
    return signal * mask


def getMyPosition(prcSoFar):
    nins, nt = prcSoFar.shape
    if nt < MIN_HIST:
        return np.zeros(nins).astype(int)

    pairs_d = pairsDollarSignal(prcSoFar)

    rsi_raw = rsiDollarSignal(prcSoFar)
    gross = np.sum(np.abs(rsi_raw))
    rsi_d = (rsi_raw / gross * OVERLAY_BUDGET) if gross > 0 else rsi_raw

    combined_dollar = pairs_d + rsi_d
    return (combined_dollar / prcSoFar[:, -1]).astype(int)
