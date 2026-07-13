"""Month-stratified, day-block bootstrap (Sprint 4.4b).

The generic significance machine for the coverage/magnitude test family
(Kupiec breach rate, Acerbi-Szekely Z1/Z2, the calibrated-vs-fhs variant
contrast). It is a RESAMPLING OBJECT, not a test statistic -- the same split
``residuals.py``/``risk.py`` drew in 4.4a, mirrored here for
``bootstrap.py``/``backtest.py``.

Two design choices, both argued in the spec (section 2.6):

- Block = delivery day. The 24 (or 23/25 on a DST edge) hours of one delivery
  day share pool, weather and regime -- resampling individual hours would tear
  that dependence apart and understate the true uncertainty. The block keeps
  a day's hours together.
- Stratum = calendar MONTH NUMBER (1-12, Europe/Berlin), pooling the SAME
  month-of-year across every year in the sample -- e.g. every June in a
  5-year backtest is one stratum, not five. Only the month-of-year
  composition is a fixed calendar effect worth holding constant; which
  specific YEAR's June contributes how many of a replicate's June days is
  deliberately left free to vary -- that is the intended randomness. Each
  replicate draws, WITHIN each month-of-year pool, exactly that pool's own
  day-count with replacement, so the total day-count per month-of-year is
  reproduced exactly every replicate, but the year mix is not.

This module deliberately does NOT import ``christoffersen_independence`` (see
``tests/test_bootstrap.py`` for the import-boundary test): the independence
family relies on the asymptotic chi-square distribution, and an i.i.d. day
resample would destroy the very serial dependence that statistic measures.
Two testing families, one resampling regime -- this module serves only one
of them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..market_time import LOCAL_TZ

log = logging.getLogger(__name__)


def _day_positions(index: pd.DatetimeIndex, *, local_tz: str) -> pd.Series:
    """Delivery day (local midnight) -> integer row positions, ascending index order."""
    local_day = index.tz_convert(local_tz).normalize()
    frame = pd.DataFrame({"day": local_day, "pos": np.arange(len(index))})
    grouped = frame.groupby("day")["pos"].apply(lambda s: s.to_numpy())
    return grouped.sort_index()


def _draw_replicate(
    month_days: dict[int, list[np.ndarray]], rng: np.random.Generator
) -> np.ndarray:
    chosen: list[np.ndarray] = []
    for days_in_month in month_days.values():
        n_days = len(days_in_month)
        draw = rng.integers(0, n_days, size=n_days)
        chosen.extend(days_in_month[d] for d in draw)
    return np.concatenate(chosen)


def stratified_day_block_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    local_tz: str = LOCAL_TZ,
    seed: int,
    ci: float = 0.95,
    min_bootstrap: int,
    max_bootstrap: int,
    check_every: int,
    mc_tol: float,
    n_stable: int,
) -> dict[str, float | int | bool]:
    """Month-of-year-stratified, day-block bootstrap CI with MC convergence monitoring.

    Stratification and blocking are UNCHANGED from the original design (spec
    2.6): stratum = calendar MONTH NUMBER (1-12) of the delivery day
    (Europe/Berlin, `local_tz`), pooling that same month across every year
    present; block = delivery day (all rows of `frame` on that day stay
    together -- pass an already valid-hours-filtered frame if that filtering
    matters to `statistic`). Each replicate draws, WITHIN each month-of-year
    pool, that pool's own day-count with replacement.

    Convergence (Nachtrag 1, part A): every `check_every` replications, this
    treats the replications drawn so far as `B // check_every` equal-length
    blocks (one block per check_every-sized chunk, in draw order) and
    estimates the Monte-Carlo standard error of BOTH reported percentiles via
    batch means -- the per-block percentile, then MCSE = sd(block
    percentiles) / sqrt(n_blocks). Stops once
    `MCSE(q_low) < mc_tol * (ci_high - ci_low)` AND
    `MCSE(q_high) < mc_tol * (ci_high - ci_low)` hold on `n_stable`
    CONSECUTIVE checkpoints -- never before `min_bootstrap`, never past
    `max_bootstrap`. Percentiles, not the mean, are monitored deliberately:
    tail-percentile variance scales roughly as `p(1-p) / (B * f(q)**2)`, and
    the density `f(q)` is small at the edges, so the reported quantities
    converge slower than the mean would.

    IMPORTANT -- "converged" does NOT mean "trustworthy". It bounds only the
    Monte-Carlo error from a finite B. It says nothing about sampling
    uncertainty from a thin stratum: a clumpy bootstrap distribution (few
    distinct days in some month-of-year pool) also converges cleanly -- to
    the percentile of a BAD approximation. `converged` NEVER replaces
    `low_support` (`cell_occupancy`); the two flags are orthogonal and both
    are meant to be reported alongside each other.

    The stopping rule reads the MC error ONLY -- it never looks at the test
    outcome (e.g. never "stop once 0 falls outside the CI"). That would be
    fishing (optional-stopping bias tied to significance).

    `statistic` maps a resampled frame to one scalar (e.g. a breach rate, a
    Z1, or a calibrated-minus-fhs difference). Returns {"point", "ci_low",
    "ci_high", "n_days", "n_bootstrap_used", "converged"}. `point` is
    `statistic(frame)` on the ORIGINAL (unresampled) data, not the bootstrap
    mean. Deterministic given `seed`, INCLUDING `n_bootstrap_used` (the draw
    sequence and the stopping rule are both deterministic functions of the
    seed). Does NOT serve the Christoffersen family (which uses chi2): an
    i.i.d. day resample would destroy the serial dependence LR_ind measures.
    """
    index = pd.DatetimeIndex(frame.index)
    day_positions = _day_positions(index, local_tz=local_tz)

    months = pd.DatetimeIndex(day_positions.index).month
    month_days: dict[int, list[np.ndarray]] = {}
    for month, positions in zip(months, day_positions, strict=True):
        month_days.setdefault(int(month), []).append(positions)

    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci
    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    thetas: list[float] = []
    stable_streak = 0
    converged = False
    next_checkpoint = check_every

    while True:
        target = min(next_checkpoint, max_bootstrap)
        while len(thetas) < target:
            replicate_positions = _draw_replicate(month_days, rng)
            thetas.append(statistic(frame.iloc[replicate_positions]))

        b = len(thetas)
        n_blocks = b // check_every
        if b >= min_bootstrap and n_blocks >= 2:
            arr = np.asarray(thetas[: n_blocks * check_every], dtype=float)
            blocks = arr.reshape(n_blocks, check_every)
            block_qlo = np.nanpercentile(blocks, lo_q, axis=1)
            block_qhi = np.nanpercentile(blocks, hi_q, axis=1)
            ci_low_now = float(np.nanpercentile(arr, lo_q))
            ci_high_now = float(np.nanpercentile(arr, hi_q))
            width = ci_high_now - ci_low_now
            mcse_lo = float(np.std(block_qlo, ddof=1) / np.sqrt(n_blocks))
            mcse_hi = float(np.std(block_qhi, ddof=1) / np.sqrt(n_blocks))
            ok = (
                (mcse_lo < mc_tol * width) and (mcse_hi < mc_tol * width)
                if width > 0
                else (mcse_lo == 0.0 and mcse_hi == 0.0)
            )
            stable_streak = stable_streak + 1 if ok else 0
            if stable_streak >= n_stable:
                converged = True
                break

        if b >= max_bootstrap:
            log.warning(
                "stratified_day_block_bootstrap did not converge within "
                "max_bootstrap=%d replications (mc_tol=%.4g, n_stable=%d); "
                "returned CI's Monte-Carlo error is not bounded as tightly as requested.",
                max_bootstrap,
                mc_tol,
                n_stable,
            )
            break
        next_checkpoint += check_every

    n_bootstrap_used = len(thetas)
    arr = np.asarray(thetas, dtype=float)
    ci_low = float(np.nanpercentile(arr, lo_q))
    ci_high = float(np.nanpercentile(arr, hi_q))
    point = float(statistic(frame))

    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_days": float(len(day_positions)),
        "n_bootstrap_used": n_bootstrap_used,
        "converged": converged,
    }


def cell_occupancy(
    frame: pd.DataFrame, *, local_tz: str = LOCAL_TZ, min_cell_days: int
) -> dict[str, object]:
    """Per-month-of-year VALID-day counts for a (already-subset) frame.

    Counts the number of distinct delivery days present in `frame` per
    calendar MONTH NUMBER (1-12), pooled across every year present -- the
    same stratum definition used by `stratified_day_block_bootstrap`. This is
    the day-block diversity the bootstrap actually draws from for THIS
    (already subset- and valid-hour-restricted) cell, regardless of whether
    any given day breached.

    (Two earlier, rejected definitions: counting BREACH-days instead of valid
    days made `min_cell_days=30` unreachable at a well-calibrated ~5% rate;
    keying strata by (year, month) instead of month-of-year made every
    February -- and any warmup-truncated boundary month, e.g. the first
    partial month after a pool warmup period -- structurally thin in EVERY
    subset, since each such month only ever has one calendar instance.
    Pooling by month-of-year across years fixes both: a "February" pool spans
    every year's February, and a partial boundary month is diluted by the
    same month-of-year's full instances in other years.)

    Flags low_support when any month-of-year pool contributes fewer than
    `min_cell_days` valid days -- the signal that a percentile CI on this
    subset is not to be trusted (spec 2.6). Returns {"low_support": bool,
    "thin_months": [...], "min_count": ...}. `thin_months` entries are the
    month numbers (1-12, as strings) below the threshold.
    """
    index = pd.DatetimeIndex(frame.index)
    local_day = index.tz_convert(local_tz).normalize()

    unique_days = pd.DatetimeIndex(sorted(pd.unique(local_day)))
    if unique_days.empty:
        return {"low_support": True, "thin_months": [], "min_count": 0.0}

    months = unique_days.month
    counts = pd.Series(1, index=months).groupby(level=0).sum()

    thin = counts[counts < min_cell_days]
    return {
        "low_support": bool(len(thin) > 0),
        "thin_months": [str(m) for m in thin.index],
        "min_count": float(counts.min()),
    }
