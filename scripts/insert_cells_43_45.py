"""Insert notebook cells for sections 4.3, 4.4, and 4.5."""

import json

NB = "notebooks/01_data_exploration.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)


def make_md(cell_id: str, lines: list[str]) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": lines}


cell_43 = make_md(
    "s4-feature-table",
    [
        "### 4.3 Feature Candidate Inventory\n",
        "\n",
        "The table below lists all candidate features for Sprint 2, organised by group."
        " The rightmost column gives the leakage assessment relative to the"
        " **gate closure** (≈ 12:00 CET on day D-1): only values known at that point"
        " may be used. The model generates a full 24-hour block forecast for day D in"
        " one shot; features requiring same-day-D observations are excluded.\n",
        "\n",
        "| Group | Candidate feature(s) | Motivating finding |"
        " Source & availability at gate closure |\n",
        "|---|---|---|---|\n",
        "| Calendar | `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`,"
        " `month_sin`, `month_cos`, `is_weekend`, `is_holiday`"
        " | Daily/weekly/annual seasonality (3.1)"
        " | Deterministic, always known. Cyclical encoding;"
        " holidays via `holidays` package (DE/LU) |\n",
        "| Forecast fundamentals | `load_forecast_day_ahead`"
        " | Price level, fishhook (3.2)"
        " | ENTSO-E day-ahead forecast, published before gate closure ✓ |\n",
        "| Forecast fundamentals | `wind_onshore_forecast`, `wind_offshore_forecast`, `solar_forecast`"
        " | Merit order, negative prices (3.2, 3.3)"
        " | ENTSO-E day-ahead renewable forecast ✓ |\n",
        "| Forecast fundamentals | `residual_load_forecast`"
        " = `load_forecast` − (`wind_onshore_forecast` + `wind_offshore_forecast` + `solar_forecast`)"
        " | Fishhook x-axis (3.2)"
        " | Derived from forecasts ✓ — **leakage-correct replacement** for the"
        " actuals-based residual load used in EDA |\n",
        "| Forecast fundamentals | `renewable_share_forecast`"
        " = (`wind_onshore_forecast` + `wind_offshore_forecast` + `solar_forecast`) / `load_forecast`"
        " | Negative-price conditions (3.3)"
        " | Derived from forecasts ✓ |\n",
        "| Commodities | `ttf_gas_eur_per_mwh`, `eua_co2_eur_per_t`"
        " | Fishhook vertical shift, spike conditions (3.4)"
        " | Daily settlement; use most recent close ≤ D-1."
        " EUA missing pre-Oct-2021 → decision D4 |\n",
        "| Lagged price | `price_lag_24h`, `price_lag_48h`, `price_lag_168h`"
        " | ACF peaks at 24h/168h (3.5)"
        " | Previous days fully known at gate closure ✓ |\n",
        "| Lagged price | `price_roll_mean_24h`, `price_roll_mean_168h`"
        " (trailing window ending ≤ gate closure)"
        " | Level/regime state (3.4) | Known ✓ |\n",
        "| Lagged price | `price_lag_1h` | Short-lag ACF (3.5)"
        " | **Conditional** — only valid for a recursive 1-step forecast,"
        " NOT a 24h-block day-ahead → decision D3 |\n",
        "| Lagged fundamentals | `residual_load_actual_lag_24h`,"
        " `load_actual_lag_24h`, `_lag_168h`"
        " | Realised demand/renewable patterns"
        " | Yesterday's actuals known ✓ (lag ≥ 24h only) |\n",
        "| Lagged cross-border | `scheduled_net_*_lag_24h`,"
        " `physical_net_*_lag_24h` (per corridor or total)"
        " | Cross-border ↔ price (3.6)"
        " | **Lag-only** — same-hour scheduled flow is endogenous (market coupling)"
        " → decision D5 |\n",
        "| Forecast-error (conditional) | `load_forecast_error_lag_24h`,"
        " `wind_onshore_forecast_error_lag_24h`,"
        " `wind_offshore_forecast_error_lag_24h`,"
        " `cross_border_deviation_lag_24h`"
        " | Forecast-error autocorrelation (4.2)"
        " | Lagged ≥ 24h ✓ — **include only if 4.2 confirms usable-lag autocorrelation**"
        " → decision D7 |\n",
        "\n",
        "**Two critical leakage corrections relative to the EDA:**\n",
        "\n",
        "**(a) Residual load and renewable share as forecast versions:**"
        " The EDA used `load_actual − gen_wind_onshore − gen_wind_offshore − gen_solar`"
        " as residual load and the corresponding actuals-based renewable share."
        " At gate closure, these actuals are not yet available for the delivery hours of day D."
        " Feature engineering must derive these quantities from the day-ahead forecasts instead.\n",
        "\n",
        "**(b) Cross-border flows and forecast errors only lagged:**"
        " Scheduled and physical cross-border flows for the delivery hours of day D are either"
        " not yet determined (physical) or endogenous to the market coupling process (scheduled)."
        " Only lagged values (≥ 24h) from previous days are leakage-free."
        " The same applies to all forecast errors: the error for hour H on day D"
        " is only observed after H has passed.",
    ],
)

cell_44 = make_md(
    "s4-open-decisions",
    [
        "### 4.4 Open Decisions for Sprint 2\n",
        "\n",
        "The following decisions must be resolved at the start of Sprint 2, before any feature"
        " code is written. Each entry states the question, options, and the owner's preliminary"
        " recommendation based on the EDA evidence.\n",
        "\n",
        "**D1 — Resolution: hourly vs 15-min.**  \n",
        "*Question:* Should the pipeline operate on hourly or 15-minute intervals?  \n",
        "*Options:* (a) Resample the post-2025-09-30 15-min data to hourly and keep a single"
        " hourly pipeline. (b) Migrate the entire pipeline to 15-min resolution.  \n",
        "*Recommendation:* Resample to hourly for the Sprint 2 MVP. The intra-hour price std is"
        " small relative to the hour-to-hour movement (analysis 4.1), so the information loss is"
        " limited. Migrating to 15-min would require resampling or re-fetching the majority of"
        " features that are published at hourly granularity by ENTSO-E."
        " Document 15-min as explicit future work.\n",
        "\n",
        "**D2 — Crisis-regime handling.**  \n",
        "*Question:* How should the 2022–2023 crisis period be treated during training?  \n",
        "*Options:* (a) Truncate the training window to exclude the crisis."
        " (b) Apply sample weighting to down-weight crisis observations."
        " (c) Include an explicit regime indicator as a feature.  \n",
        "*Recommendation:* Include TTF gas price as a feature (already in the inventory);"
        " a tree model (LightGBM) can split on it directly to capture the fishhook vertical shift."
        " A hard regime dummy is likely redundant once TTF is in the model — verify by experiment."
        " Start without a regime dummy; add it only if residuals show unexplained regime dependence.\n",
        "\n",
        "**D3 — Block vs recursive forecast and the 1h lag.**  \n",
        "*Question:* Should the model produce a 24-hour block forecast or a recursive"
        " 1-step-ahead forecast?  \n",
        "*Options:* (a) Block forecast (one model, 24 target hours simultaneously or separate per hour)."
        " (b) Recursive: forecast hour by hour, feeding predicted prices as inputs.  \n",
        "*Recommendation:* Block forecast (day-ahead, as the real market operates)."
        " Consequence: `price_lag_1h` is leakage for a block model targeting all 24 hours of day D"
        " — the previous hour on day D is not available at gate closure. Use lags ≥ 24h only."
        " This also simplifies the backtesting walk-forward setup"
        " (one prediction per day, not 24 sequential steps).\n",
        "\n",
        "**D4 — EUA CO2 missingness before Oct-2021 (~38% of training data).**  \n",
        "*Question:* How to handle the ~38% NaN period for EUA CO2?  \n",
        "*Options:* (a) Truncate training to ≥ Oct-2021 (loses the “calm” regime)."
        " (b) Keep full range with a `eua_missing` indicator flag."
        " (c) Source EUA data from EEX/ICE for the pre-2021 period.  \n",
        "*Recommendation:* Start with option (b): include a `eua_missing` binary flag so the model"
        " can learn a different price equation for the pre-2021 window. If the model fails to learn"
        " a useful representation of the calm regime, switch to (a). Option (c) is the"
        " production-grade solution but out of scope for the portfolio project.\n",
        "\n",
        "**D5 — Cross-border features: lag only.**  \n",
        "*Fixed:* Scheduled and physical cross-border flows for the delivery hours of day D must"
        " only be used with lag ≥ 24h. Same-hour values are endogenous (market coupling) and"
        " constitute leakage. This applies to both scheduled and physical flows for all six corridors.\n",
        "\n",
        "**D6 — Target transformation.**  \n",
        "*Question:* Should the day-ahead price be transformed before modelling?  \n",
        "*Options:* (a) No transformation (raw EUR/MWh). (b) `asinh` (defined for negative values"
        " and zero, log-like for large positives). (c) Signed log: `sign(x) * log(1 + |x|)`."
        " (d) Quantile/rank transform.  \n",
        "*Recommendation:* No transformation for LightGBM (tree models are scale-invariant and"
        " handle skew natively). For an ARIMAX or linear baseline, consider `asinh` given the"
        " negative prices. For quantile regression (Pinball loss), no transformation is needed."
        " Decide per model type in Sprint 2/3.\n",
        "\n",
        "**D7 — Forecast-error features: include conditionally.**  \n",
        "*Question:* Which lagged forecast-error features should be included?  \n",
        "*Evidence from 4.2:* Load error — clear persistence at lag 24h/168h → **include**."
        " Wind onshore error — clear persistence → **include**."
        " Wind offshore error — moderate persistence → **include, verify with SHAP**."
        " Solar error — plausible persistence (daytime lags) → **include conditionally**."
        " Cross-border deviation — weaker signal → **include, drop if SHAP confirms no contribution**.  \n",
        "*Recommendation:* Include all five as candidates in the initial feature set;"
        " prune via SHAP importance in Sprint 3.",
    ],
)

cell_45 = make_md(
    "s4-summary",
    [
        "### 4.5 Summary and Hand-off to Feature Engineering\n",
        "\n",
        "**Data quality (section 2):** The DE-LU dataset is structurally sound."
        " The main data-quality constraints that feature engineering must handle are:"
        " the mixed hourly/15-min resolution break at 2025-09-30 (→ D1),"
        " EUA CO2 missingness before Oct-2021 (→ D4),"
        " and the permanent disappearance of `gen_nuclear` after April 2023"
        " (treat as a known structural zero). All other ENTSO-E columns are >99% complete.\n",
        "\n",
        "**Market structure (section 3):** The day-ahead price is driven by three interacting"
        " mechanisms: (1) residual load (the physical supply-demand balance after renewables,"
        " setting which plant is marginal), (2) the gas price (TTF), which shifts the entire"
        " merit-order curve vertically, and (3) calendar effects (daily 24h and weekly 168h cycles)."
        " Negative prices are systematic, not noise — driven by high renewable share at low demand,"
        " growing year-on-year. The 2022 crisis constitutes a distinct regime in both level and"
        " volatility; the gas price feature should allow a tree model to capture this without a"
        " hard regime dummy.\n",
        "\n",
        "**Targeted analyses (section 4):** Intra-hour price variation in the 15-min era is small"
        " relative to hour-to-hour dynamics, supporting a hourly resampling decision (D1)."
        " Forecast errors for load, wind onshore, and wind offshore show meaningful autocorrelation"
        " at actionable lags (≥ 24h), justifying dedicated lagged error features (D7).\n",
        "\n",
        "**Primary hand-off artefact:** The feature candidate inventory in 4.3 lists all candidate"
        " features with their leakage status relative to gate closure. This table is the direct"
        " input for `src/energy_price_forecast/features/` in Sprint 2. The two critical leakage"
        " corrections to carry forward are: (a) use *forecast* versions of residual load and"
        " renewable share, not actuals; (b) use cross-border flows and forecast errors"
        " *lagged ≥ 24h only*.\n",
        "\n",
        "**Open decisions (4.4):** D1–D7 must be resolved at the start of Sprint 2 before any"
        " feature code is written. This notebook is analysis only — no production features,"
        " pipeline code, or models have been built here.",
    ],
)

# Insert after anchor
anchor = "7dc5490c"
idx = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == anchor)
# Insert in reverse order so final sequence is: anchor, 4.3, 4.4, 4.5
nb["cells"].insert(idx + 1, cell_45)
nb["cells"].insert(idx + 1, cell_44)
nb["cells"].insert(idx + 1, cell_43)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Total cells: {len(nb['cells'])}")
print("Last 4 cell IDs:")
for cell in nb["cells"][-4:]:
    print(" ", cell.get("id"), "|", (cell["source"][0] if cell["source"] else "")[:60])
