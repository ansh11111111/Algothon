import numpy as np

nInst = 51

# ---- Strategy parameters (tuned via grid search in sweep.py / strategies.py) ----
Z_LOOKBACK = 40        # days of spread history used to compute the entry z-score
DOLLAR_BUDGET = 8000    # controls how large each pair's positions get before eval.py's own $ clip
MIN_HIST = Z_LOOKBACK + 2

# ---- Pairs trading universe ----

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


def getMyPosition(prcSoFar):

    nins, nt = prcSoFar.shape
    pos = np.zeros(nins)


    if nt < MIN_HIST:
        return pos.astype(int)

    for p in PAIRS:
        i, j, beta, alpha = p['i'], p['j'], p['beta'], p['alpha']


        price_i_hist = prcSoFar[i, :]
        price_j_hist = prcSoFar[j, :]
        spread_hist = price_i_hist - beta * price_j_hist - alpha


        recent = spread_hist[-Z_LOOKBACK:]
        mu = recent.mean()
        sd = recent.std()
        if sd < 1e-8:
            continue
        z = (spread_hist[-1] - mu) / sd


        signal = -z


        price_i = prcSoFar[i, -1]
        pos_i_shares = signal * DOLLAR_BUDGET / price_i
        pos_j_shares = -beta * pos_i_shares

        pos[i] += pos_i_shares
        pos[j] += pos_j_shares


    return np.array([int(x) for x in pos])
