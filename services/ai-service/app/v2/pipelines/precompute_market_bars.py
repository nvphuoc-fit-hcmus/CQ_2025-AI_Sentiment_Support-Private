"""Precompute per-timeframe bar sequences for SAFE-Alert paper Eq.16.

Session 23 P0 #3: SAFE-Alert Section 3.5, Eq.16 specifies
    X^(δ) ∈ R^{L_δ × d_δ}
    = sequence of L_δ most-recent bars at timeframe δ, d_δ features per bar
for δ ∈ {1m, 5m, 15m, 1h, 4h}.

The previous `precompute_market_features.py` produced 63-dim SCALAR aggregates
per candle, which is a simplification (each timeframe collapsed to one vector).
This script produces the paper-literal bar-sequence representation so the
upgraded `MultiTimescaleMarketEncoder(use_bar_sequences=True)` can consume
genuine temporal sequences and realise Contribution 3 of the paper.

Output
------
``market_bars.npz`` for the default 1h decision horizon, or
``market_bars_{decision_horizon}.npz`` for other horizons, with 5 float32 arrays:

    bars_1m  : (N_candles, L, d)
    bars_5m  : (N_candles, L, d)
    bars_15m : (N_candles, L, d)
    bars_1h  : (N_candles, L, d)
    bars_4h  : (N_candles, L, d)

where ``N_candles`` = number of decision marks for the requested horizon
(same order as ``BTCUSDT_{decision_horizon}_ohlcv.csv``), ``L`` = 20 (bar-sequence length, paper unspecified
— 20 is the standard TA window), ``d`` = 10 (per-bar features).

Per-bar features (10 dims)
---------------------------
0. log(open)          — raw price level (log-scaled)
1. log(high)
2. log(low)
3. log(close)
4. log(volume + 1)    — +1 avoids log(0)
5. log_return         = log(close / open)
6. body               = (close - open) / open     — normalised body size
7. upper_wick         = (high - max(open, close)) / open
8. lower_wick         = (min(open, close) - low)  / open
9. volatility_pct     = (high - low) / open       — range as % of open

Causality
---------
For each decision mark t (a horizon candle close time), the last ``L`` bars at
timeframe δ are those with ``bar_close_time <= t``. No future bars leak in.
Since market exchanges provide bars immediately on close, no ingest-delay
subtraction is needed for market data (unlike news — see ingest_delay_minutes
on the SAFEAlertDataset). If fewer than ``L`` historical bars exist (early
history of the dataset), the sequence is left-padded with zeros and the
first valid candle index is tracked.

Usage
-----
    python precompute_market_bars.py \
        --data-dir training_data/v2 \
        --symbol BTCUSDT \
        --decision-horizon 1h \
        --bar-seq-len 20

Then train with ``use_bar_sequences: true`` in train_config_research_best.yaml.

Reproducibility
---------------
This script is deterministic: given the same CSVs and args, it emits the
same .npz file. No randomness is involved in feature computation.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("precompute_market_bars")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Paper spec constants (see module docstring)
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAMES: List[str] = ["1m", "5m", "15m", "1h", "4h"]   # δ ∈ paper Eq.16
BAR_FEAT_DIM: int = 10                                     # d_δ per bar
DEFAULT_BAR_SEQ_LEN: int = 20                              # L_δ


def _read_ohlcv(path: Path) -> pd.DataFrame:
    """Load an OHLCV CSV and normalize the schema.

    Expected columns: datetime, open, high, low, close, volume.
    Returns a DataFrame with a UTC tz-aware ``datetime`` column sorted
    ascending by time, duplicates dropped.
    """
    if not path.exists():
        raise FileNotFoundError(f"OHLCV CSV not found: {path}")
    df = pd.read_csv(path)
    # Normalise 'datetime' column name — some dumps use 'timestamp' etc.
    col_map = {c.lower(): c for c in df.columns}
    dt_col = (
        col_map.get("datetime")
        or col_map.get("timestamp")
        or col_map.get("time")
        or col_map.get("date")
    )
    if dt_col is None:
        raise ValueError(f"{path.name}: no datetime column (tried datetime/timestamp/time/date)")
    df = df.rename(columns={dt_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(
        subset=["datetime"], keep="last"
    ).reset_index(drop=True)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - {c.lower() for c in df.columns}
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    # Ensure lowercase numeric cols
    for c in required:
        if c not in df.columns:
            actual = [col for col in df.columns if col.lower() == c]
            df = df.rename(columns={actual[0]: c})
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def _compute_bar_features(bar_window: pd.DataFrame) -> np.ndarray:
    """Convert an L-row OHLCV window into an (L, 10) feature array.

    See module docstring for the exact feature list.
    """
    o = bar_window["open"].to_numpy(dtype=np.float64)
    h = bar_window["high"].to_numpy(dtype=np.float64)
    l = bar_window["low"].to_numpy(dtype=np.float64)
    c = bar_window["close"].to_numpy(dtype=np.float64)
    v = bar_window["volume"].to_numpy(dtype=np.float64)
    # Numerical safety: guard against zero / negative open (shouldn't happen
    # on real exchange data but defensive). Single `_safe` convention used
    # consistently below — replaces previous mix of np.where / np.maximum
    # which was functionally equivalent but confusing.
    o_safe = np.maximum(o, 1e-8)
    h_safe = np.maximum(h, 1e-8)
    l_safe = np.maximum(l, 1e-8)
    c_safe = np.maximum(c, 1e-8)
    feats = np.zeros((len(o), BAR_FEAT_DIM), dtype=np.float32)
    feats[:, 0] = np.log(o_safe)
    feats[:, 1] = np.log(h_safe)
    feats[:, 2] = np.log(l_safe)
    feats[:, 3] = np.log(c_safe)
    feats[:, 4] = np.log1p(np.maximum(v, 0.0))
    feats[:, 5] = np.log(c_safe / o_safe)
    feats[:, 6] = (c - o) / o_safe
    feats[:, 7] = (h - np.maximum(o, c)) / o_safe
    feats[:, 8] = (np.minimum(o, c) - l) / o_safe
    feats[:, 9] = (h - l) / o_safe
    # Clip extreme outliers (pathological fat-finger bars) so downstream
    # normalisation stays well-conditioned.
    np.clip(feats[:, 5:], -1.0, 1.0, out=feats[:, 5:])
    return feats


def _infer_interval(times: pd.Series | pd.DatetimeIndex, fallback: str = "1h") -> np.timedelta64:
    """Infer a candle/bar interval from sorted timestamps."""
    idx = pd.DatetimeIndex(times)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    values = idx.values.astype("datetime64[ns]")
    if len(values) >= 2:
        diffs = np.diff(values).astype("timedelta64[ns]").astype(np.int64)
        interval_ns = int(np.median(diffs))
        return np.timedelta64(max(interval_ns, 1), "ns")
    return np.timedelta64(int(pd.Timedelta(fallback).value), "ns")


def _extract_sequences_for_timeframe(
    decision_times: pd.DatetimeIndex,
    decision_interval: np.timedelta64,
    ohlcv: pd.DataFrame,
    seq_len: int,
    tf_name: str,
) -> np.ndarray:
    """For each decision-mark timestamp t, return the last ``seq_len`` bars
    at timeframe δ with ``bar_close_time <= t + decision_interval``. Output shape
    ``(N_decisions, seq_len, BAR_FEAT_DIM)``.

    Left-padded with zeros when fewer than ``seq_len`` bars are available
    (cold-start of the dataset).

    Causality (corrected in Session 24 final audit):
    -----------------------------------------------
    Binance convention: bar ``datetime`` is the OPEN time; the bar covers
    ``[datetime, datetime + interval)`` and closes at ``datetime + interval``.
    Decision marks come from ``BTCUSDT_1h_ohlcv.csv`` (1h cadence) and the
    dataset's scalar features include the current 1h candle's OHLCV — so
    the effective decision time is ``candle_datetime + decision_interval``
    (moment the current decision candle closes).

    Previous implementation used ``searchsorted(bar_open_times, candle_datetime,
    side='right')`` which leaked coarser-TF bars: a 4h bar opening at
    ``candle_datetime`` has close_time = ``candle_datetime + 4h`` — 3h of
    future price action baked into the sequence.

    Fix: compute per-TF ``close_times = bar_open_times + interval`` and
    searchsort against those, with effective decision =
    ``candle_datetime + decision_interval``. Side="right" on close_times includes bars
    whose close is at or before the decision time, excludes future bars.
    """
    # Sorted bar open/close times and precomputed features.
    # Pandas 2.x Timestamp with tz converts to dtype=object when .to_numpy()
    # is called — breaks datetime64 arithmetic (bar_times + interval) and
    # searchsorted (comparison between Timestamp and int64). We go through
    # pd.DatetimeIndex to guarantee tz-naive datetime64[ns] output.
    bar_idx = pd.DatetimeIndex(ohlcv["datetime"])
    if bar_idx.tz is not None:
        bar_idx = bar_idx.tz_convert("UTC").tz_localize(None)
    bar_times = bar_idx.values.astype("datetime64[ns]")
    all_feats = _compute_bar_features(ohlcv)   # (N_bars_total, 10)

    # Infer per-TF interval from the median gap (robust against occasional
    # missing bars / weekend gaps). Fall back to 1h if only one bar.
    interval = _infer_interval(pd.DatetimeIndex(bar_times), fallback="1h")

    # close_time[k] = open_time[k] + interval
    close_times = bar_times + interval   # (N_bars_total,)

    # Effective decision time = candle_datetime + decision_interval.
    # This matches the dataset's convention where the current candle's OHLC
    # is considered "just closed" for feature extraction.
    # Same tz-naive datetime64[ns] treatment as bar_times to guarantee
    # searchsorted works on matching dtypes.
    dec_idx = pd.DatetimeIndex(decision_times) + pd.Timedelta(decision_interval)
    if dec_idx.tz is not None:
        dec_idx = dec_idx.tz_convert("UTC").tz_localize(None)
    dec_np = dec_idx.values.astype("datetime64[ns]")

    # For each decision mark t', find first bar with close_time > t'.
    # Bars [0:cut_idx] have close_time ≤ t' → fully closed, safe to use.
    cut_idx = np.searchsorted(close_times, dec_np, side="right")  # (N_decisions,)

    N = len(dec_np)
    out = np.zeros((N, seq_len, BAR_FEAT_DIM), dtype=np.float32)
    cold_start = 0
    for i in range(N):
        end = int(cut_idx[i])
        start = max(0, end - seq_len)
        take = end - start
        if take <= 0:
            cold_start += 1
            continue
        out[i, seq_len - take: seq_len] = all_feats[start:end]
    logger.info(
        "  %s: N_decisions=%d, total_bars=%d, interval=%s, cold-start=%d",
        tf_name, N, len(ohlcv), pd.Timedelta(interval), cold_start,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path,
                        default=Path("training_data/v2"),
                        help="Directory containing {SYMBOL}_{TF}_ohlcv.csv files")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--decision-horizon", type=str, default="1h",
                        choices=["15m", "1h", "4h", "24h"],
                        help="Decision horizon whose candle closes define N (default: 1h)")
    parser.add_argument("--bar-seq-len", type=int, default=DEFAULT_BAR_SEQ_LEN,
                        help="L_δ per timeframe (default: 20)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output .npz path (default: market_bars.npz for 1h, market_bars_{horizon}.npz otherwise)")
    args = parser.parse_args()

    if args.out is not None:
        out_path = args.out
    elif args.decision_horizon == "1h":
        out_path = args.data_dir / "market_bars.npz"
    else:
        out_path = args.data_dir / f"market_bars_{args.decision_horizon}.npz"

    # ── Decision marks come from the 1h CSV (matches training horizon=1h) ──
    # Each requested-horizon candle close_time is one decision mark t.
    # For non-1h canaries this selects BTCUSDT_4h_ohlcv.csv, etc.
    decision_path = args.data_dir / f"{args.symbol}_{args.decision_horizon}_ohlcv.csv"
    if not decision_path.exists():
        raise FileNotFoundError(
            f"{args.decision_horizon} OHLCV CSV not found: {decision_path}. "
            f"Decision marks are derived from this file; it must exist."
        )
    decision_df = _read_ohlcv(decision_path)
    decision_times = decision_df["datetime"]
    decision_interval = _infer_interval(pd.DatetimeIndex(decision_times), fallback=args.decision_horizon)
    N = len(decision_times)
    logger.info(
        "Decision marks: %d (from %s, interval=%s)",
        N, decision_path.name, pd.Timedelta(decision_interval),
    )

    # ── Per-timeframe extraction ──
    out_arrays = {}
    for tf in TIMEFRAMES:
        tf_path = args.data_dir / f"{args.symbol}_{tf}_ohlcv.csv"
        logger.info("Processing %s (%s)", tf, tf_path.name)
        tf_df = _read_ohlcv(tf_path)
        seqs = _extract_sequences_for_timeframe(
            decision_times=pd.DatetimeIndex(decision_times),
            decision_interval=decision_interval,
            ohlcv=tf_df,
            seq_len=args.bar_seq_len,
            tf_name=tf,
        )
        out_arrays[f"bars_{tf}"] = seqs

    # ── Sanity checks ──
    for key, arr in out_arrays.items():
        assert arr.shape == (N, args.bar_seq_len, BAR_FEAT_DIM), (
            f"{key} shape {arr.shape} != expected ({N}, {args.bar_seq_len}, {BAR_FEAT_DIM})"
        )
        assert np.isfinite(arr).all(), f"{key} contains NaN/Inf"

    # Persist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **out_arrays,
        decision_horizon=np.array(args.decision_horizon),
        decision_interval_ns=np.array(pd.Timedelta(decision_interval).value, dtype=np.int64),
        bar_seq_len=np.array(args.bar_seq_len, dtype=np.int32),
        bar_feat_dim=np.array(BAR_FEAT_DIM, dtype=np.int32),
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %s (%.1f MB)", out_path, size_mb)
    logger.info("Shape per TF: (%d, %d, %d)", N, args.bar_seq_len, BAR_FEAT_DIM)
    logger.info("Ready to train with use_bar_sequences=true")


if __name__ == "__main__":
    main()
