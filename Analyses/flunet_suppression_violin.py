from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PATHOGEN_SPECS = [
    {"key": "RSV", "label": "RSV", "column": "RSV"},
    {"key": "Metapneumovirus", "label": "Metapneumovirus", "column": "METAPNEUMO"},
    {"key": "Parainfluenza", "label": "Parainfluenza", "column": "PARAINFLUENZA"},
    {"key": "Adenovirus", "label": "Adenovirus", "column": "ADENO"},
    {"key": "InfluenzaA", "label": "InfluenzaA", "column": "INF_A"},
    {"key": "InfluenzaB", "label": "InfluenzaB", "column": "INF_B"},
]

PATHOGEN_COLUMN_BY_KEY = {spec["key"]: spec["column"] for spec in PATHOGEN_SPECS}
SPEC_PROCESSED_COLUMN = "SPEC_PROCESSED_NB"
DQ_ZERO_RUN_SETS = [
    ("set3", {"Adenovirus", "Parainfluenza", "Metapneumovirus"}),
    ("set4", {"Adenovirus", "Parainfluenza", "Metapneumovirus", "RSV"}),
    ("set6", {"Adenovirus", "Parainfluenza", "Metapneumovirus", "RSV", "InfluenzaA", "InfluenzaB"}),
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compute country-level COVID suppression durations from FluNet and plot "
            "pathogen distributions as violins."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "Data/Raw/FluNet.csv",
        help="Path to FluNet.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "Data/Processed/FluNet_suppression_duration_by_country_pathogen.csv",
        help="Path to write per country-pathogen suppression results",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=repo_root / "Data/Processed/FluNet_suppression_duration_pathogen_summary.csv",
        help="Path to write pathogen-level summary statistics",
    )
    parser.add_argument(
        "--output-dq-csv",
        type=Path,
        default=repo_root / "Data/Processed/FluNet_country_data_quality_diagnostics.csv",
        help="Path to write country-level data-quality diagnostics",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=repo_root / "Figures/FluNet_suppression_duration_violin.png",
        help="Path to save violin plot",
    )
    parser.add_argument(
        "--output-example-figure",
        type=Path,
        default=repo_root / "Figures/FluNet_suppression_examples_10_countries.png",
        help="Path to save pathogen-country time-series example plot",
    )
    parser.add_argument(
        "--output-timeseries-dir",
        type=Path,
        default=repo_root / "Data/Processed/FluNetTimeseries",
        help="Directory to write one monthly time-series CSV per country-pathogen",
    )
    parser.add_argument(
        "--anchor-date",
        type=str,
        default="2020-03-01",
        help="Suppression search anchor date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--threshold-divisor",
        type=float,
        default=20.0,
        help="Threshold divisor used in suppression definition (max/divisor)",
    )
    parser.add_argument(
        "--pre2020-monitoring-threshold",
        type=float,
        default=30.0,
        help=(
            "Pre-2020 monthly count used to mark start of reliable monitoring for baseline-gap comparison"
        ),
    )
    parser.add_argument(
        "--pre2020-max-lookback-years",
        type=int,
        default=5,
        help="Maximum pre-2020 lookback window (years) used for baseline-gap comparison",
    )
    parser.add_argument(
        "--pre2020-baseline-gap-cap-months",
        type=int,
        default=12,
        help="Cap on pre-2020 longest below-threshold run used for excess-gap exclusion",
    )
    parser.add_argument(
        "--min-months",
        type=int,
        default=12,
        help="Minimum observed months required per country-pathogen",
    )
    parser.add_argument(
        "--dq-zero-run-months",
        type=int,
        default=24,
        help="Exclude triggered sets when longest contiguous all-zero run is strictly greater than this value",
    )
    parser.add_argument(
        "--dq-low-activity-max-monthly",
        type=float,
        default=30.0,
        help="Exclude pathogen-country series if all monthly counts are below this threshold",
    )
    parser.add_argument(
        "--dq-pre2020-min-max-monthly",
        type=float,
        default=0.0,
        help="Exclude pathogen-country series if pre-2020 maximum monthly count is below this threshold",
    )
    parser.add_argument(
        "--dq-post-start-date",
        type=str,
        default="2020-03-01",
        help="Date from which to apply post-period data-quality checks (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--dq-post-min-nonzero-months",
        type=int,
        default=3,
        help="Minimum number of post-start months with non-zero counts required per pathogen-country",
    )
    parser.add_argument(
        "--n-example-countries",
        type=int,
        default=10,
        help="Number of random example countries for time-series panel plot",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=260604,
        help="Random seed used to sample example countries",
    )
    parser.add_argument(
        "--allow-partial-example-countries",
        action="store_true",
        help="Allow example countries even if not all six pathogens pass inclusion tests",
    )
    parser.add_argument(
        "--example-country-pathogen",
        type=str,
        default=None,
        help=(
            "Optional pathogen filter for example-country sampling. "
            "Use one of: RSV, Metapneumovirus, Parainfluenza, Adenovirus, InfluenzaA, InfluenzaB; "
            "or pass none/None/null to disable."
        ),
    )
    args = parser.parse_args()

    if args.example_country_pathogen is not None:
        token = args.example_country_pathogen.strip()
        token_lower = token.lower()
        canonical_by_lower = {k.lower(): k for k in PATHOGEN_COLUMN_BY_KEY}
        legacy_aliases = {
            "rsv": "RSV",
            "hmpv": "Metapneumovirus",
            "piv": "Parainfluenza",
            "adv": "Adenovirus",
            "flu_a": "InfluenzaA",
            "flu_b": "InfluenzaB",
        }
        if token_lower in {"", "none", "null"}:
            args.example_country_pathogen = None
        elif token_lower in canonical_by_lower:
            args.example_country_pathogen = canonical_by_lower[token_lower]
        elif token_lower in legacy_aliases:
            args.example_country_pathogen = legacy_aliases[token_lower]
        else:
            valid = ", ".join(sorted(PATHOGEN_COLUMN_BY_KEY))
            parser.error(
                f"Invalid --example-country-pathogen '{args.example_country_pathogen}'. "
                f"Use one of: {valid}, or none."
            )

    return args


def load_and_aggregate_monthly(input_path: Path) -> pd.DataFrame:
    keep_cols = ["COUNTRY_AREA_TERRITORY", "ISO_WEEKSTARTDATE", SPEC_PROCESSED_COLUMN] + [
        spec["column"] for spec in PATHOGEN_SPECS
    ]
    df = pd.read_csv(input_path, usecols=keep_cols).copy()
    df = df.assign(ISO_WEEKSTARTDATE=pd.to_datetime(df["ISO_WEEKSTARTDATE"], errors="coerce"))
    df = df.dropna(subset=["ISO_WEEKSTARTDATE", "COUNTRY_AREA_TERRITORY"]).copy()

    for spec in PATHOGEN_SPECS:
        df.loc[:, spec["column"]] = pd.to_numeric(df[spec["column"]], errors="coerce").fillna(0.0)
    df.loc[:, SPEC_PROCESSED_COLUMN] = pd.to_numeric(df[SPEC_PROCESSED_COLUMN], errors="coerce").fillna(0.0)

    df = df.assign(month_start=df["ISO_WEEKSTARTDATE"].dt.to_period("M").dt.to_timestamp())
    monthly = (
        df.groupby(["COUNTRY_AREA_TERRITORY", "month_start"], as_index=False)[
            [SPEC_PROCESSED_COLUMN] + [spec["column"] for spec in PATHOGEN_SPECS]
        ]
        .sum()
        .sort_values(["COUNTRY_AREA_TERRITORY", "month_start"])
    )
    return monthly


def month_diff(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def empty_exclusion(reason: str, series_end: pd.Timestamp | None) -> dict:
    return {
        "status": "excluded",
        "reason": reason,
        "threshold": np.nan,
        "dip_time": pd.NaT,
        "last_pre_time": pd.NaT,
        "rebound_time": pd.NaT,
        "suppression_duration_months": np.nan,
        "is_right_censored": False,
        "series_end": series_end,
    }


def longest_zero_run_info(series: pd.Series) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    zero_mask = series.eq(0)
    if not bool(zero_mask.any()):
        return 0, pd.NaT, pd.NaT

    best_len = 0
    best_start = pd.NaT
    best_end = pd.NaT
    current_len = 0
    current_start = pd.NaT

    for idx, is_zero in zero_mask.items():
        if bool(is_zero):
            if current_len == 0:
                current_start = idx
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
                best_end = idx
        else:
            current_len = 0

    return int(best_len), best_start, best_end


def longest_below_threshold_run_months(series: pd.Series, threshold: float) -> int:
    below_mask = series.lt(threshold)
    if not bool(below_mask.any()):
        return 0

    best_len = 0
    current_len = 0
    for is_below in below_mask.values:
        if bool(is_below):
            current_len += 1
            if current_len > best_len:
                best_len = current_len
        else:
            current_len = 0
    return int(best_len)


def longest_zero_run_within_window(series: pd.Series, start_time: pd.Timestamp, end_time: pd.Timestamp) -> int:
    window = series[(series.index > start_time) & (series.index <= end_time)]
    if window.empty:
        return 0
    return longest_zero_run_info(window)[0]


def build_country_data_quality_diagnostics(
    monthly_df: pd.DataFrame,
    min_months: int,
    dq_zero_run_months: int,
    dq_low_activity_max_monthly: float,
    dq_pre2020_min_max_monthly: float,
    dq_post_start_date: pd.Timestamp,
    dq_post_min_nonzero_months: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    for country, country_df in monthly_df.groupby("COUNTRY_AREA_TERRITORY"):
        country_df = country_df.set_index("month_start").sort_index()
        full_month_idx = pd.date_range(country_df.index.min(), country_df.index.max(), freq="MS")

        reindexed = {}
        for spec in PATHOGEN_SPECS:
            col = spec["column"]
            reindexed[col] = country_df[col].reindex(full_month_idx, fill_value=0.0)

        row: dict = {
            "country": country,
            "n_months": int(len(full_month_idx)),
            "dq_insufficient_months": bool(len(full_month_idx) < min_months),
            "dq_zero_run_threshold_months": int(dq_zero_run_months),
            "dq_low_activity_threshold": float(dq_low_activity_max_monthly),
            "dq_pre2020_min_max_monthly": float(dq_pre2020_min_max_monthly),
            "dq_post_start_date": dq_post_start_date,
            "dq_post_min_nonzero_months": int(dq_post_min_nonzero_months),
        }

        post_mask = full_month_idx >= dq_post_start_date
        pre2020_mask = full_month_idx < pd.Timestamp("2020-03-01")
        post_month_idx = full_month_idx[post_mask]
        row["dq_post_n_months"] = int(len(post_month_idx))

        for spec in PATHOGEN_SPECS:
            key = spec["key"]
            col = spec["column"]
            s = reindexed[col]
            row[f"{key}_max_monthly"] = float(s.max())
            row[f"{key}_months_ge_low_activity_threshold"] = int((s >= dq_low_activity_max_monthly).sum())
            row[f"{key}_all_zero"] = bool(float(s.max()) <= 0)
            row[f"{key}_all_months_below_low_activity_threshold"] = bool(float(s.max()) < dq_low_activity_max_monthly)
            pre_s = s[pre2020_mask]
            pre_max = float(pre_s.max()) if len(pre_s) else 0.0
            row[f"{key}_pre2020_max_monthly"] = pre_max
            row[f"{key}_pre2020_meets_min_max_monthly"] = bool(pre_max >= dq_pre2020_min_max_monthly)
            s_post = s[post_mask]
            row[f"{key}_post_nonzero_months"] = int((s_post > 0).sum())
            row[f"{key}_post_meets_min_nonzero_months"] = bool(int((s_post > 0).sum()) >= dq_post_min_nonzero_months)

        for set_name, keys in DQ_ZERO_RUN_SETS:
            cols = [PATHOGEN_COLUMN_BY_KEY[k] for k in keys]
            set_sum = sum(reindexed[c] for c in cols)
            set_sum = set_sum[post_mask]
            run_len, run_start, run_end = longest_zero_run_info(set_sum)
            row[f"{set_name}_longest_zero_run_months"] = int(run_len)
            row[f"{set_name}_longest_zero_run_start"] = run_start
            row[f"{set_name}_longest_zero_run_end"] = run_end
            row[f"{set_name}_trigger"] = bool(run_len > dq_zero_run_months)

        rows.append(row)

    return pd.DataFrame(rows)


def get_dq_exclusion_reason(
    series: pd.Series,
    pathogen_key: str,
    country_diag: pd.Series,
    dq_low_activity_max_monthly: float,
    dq_pre2020_min_max_monthly: float,
    dq_post_min_nonzero_months: int,
) -> tuple[bool, str, str, int]:
    max_value = float(series.max()) if len(series) else np.nan
    months_above = int((series >= dq_low_activity_max_monthly).sum()) if len(series) else 0

    if max_value <= 0:
        return True, "all_zero", "none", 0

    pre2020_max = float(country_diag.get(f"{pathogen_key}_pre2020_max_monthly", 0.0))
    if pre2020_max < dq_pre2020_min_max_monthly:
        return True, "low_pre2020_max_monthly", "none", months_above

    if max_value < dq_low_activity_max_monthly:
        return True, "low_activity_all_months_below_threshold", "none", months_above

    post_nonzero_months = int(country_diag.get(f"{pathogen_key}_post_nonzero_months", 0))
    if post_nonzero_months < dq_post_min_nonzero_months:
        return True, "insufficient_post_nonzero_months", "none", months_above

    return False, "", "none", months_above


def evaluate_suppression(
    series: pd.Series,
    processed_series: pd.Series,
    anchor_date: pd.Timestamp,
    threshold_divisor: float,
    min_months: int,
    pre2020_monitoring_threshold: float,
    pre2020_max_lookback_years: int,
    pre2020_baseline_gap_cap_months: int,
) -> dict:
    series = series.sort_index()
    if len(series) < min_months:
        return empty_exclusion("insufficient_months", series.index.max() if len(series) else pd.NaT)

    max_value = float(series.max())
    if max_value <= 0:
        return empty_exclusion("all_zero", series.index.max())

    pre_2020 = series[series.index < pd.Timestamp("2020-03-01")]
    if pre_2020.empty:
        return empty_exclusion("insufficient_pre2020_baseline", series.index.max())

    pre_2020_max = float(pre_2020.max())
    if pre_2020_max <= 0:
        return empty_exclusion("no_pre2020_signal", series.index.max())

    threshold = pre_2020_max / threshold_divisor
    pre2020_end = pd.Timestamp("2020-03-01")
    lookback_start = pre2020_end - pd.DateOffset(years=pre2020_max_lookback_years)
    above_monitoring = pre_2020[pre_2020 > pre2020_monitoring_threshold]
    if not above_monitoring.empty:
        first_above_monitoring = above_monitoring.index.min()
        baseline_start = max(lookback_start, first_above_monitoring)
    else:
        baseline_start = lookback_start

    pre_2020_window = pre_2020[pre_2020.index >= baseline_start]
    if pre_2020_window.empty:
        pre_2020_window = pre_2020

    pre2020_longest_below_threshold_run = longest_below_threshold_run_months(pre_2020_window, threshold)
    pre2020_effective_baseline_run = min(pre2020_longest_below_threshold_run, pre2020_baseline_gap_cap_months)

    post_anchor = series[series.index > anchor_date]
    dip_candidates = post_anchor[post_anchor < threshold]
    if dip_candidates.empty:
        out = empty_exclusion("no_dip_after_anchor", series.index.max())
        out["threshold"] = threshold
        out["pre2020_baseline_start"] = baseline_start
        out["pre2020_longest_below_threshold_run_months"] = pre2020_longest_below_threshold_run
        out["pre2020_effective_baseline_run_months"] = pre2020_effective_baseline_run
        return out

    dip_time = dip_candidates.index.min()
    dip_idx = int(series.index.get_loc(dip_time))
    last_pre_time = series.index[max(dip_idx - 1, 0)]

    rebound_candidates = series[(series.index > last_pre_time) & (series > threshold)]
    if rebound_candidates.empty:
        series_end = series.index.max()
        duration_months = month_diff(last_pre_time, series_end)
        excess_gap_months = max(duration_months - pre2020_effective_baseline_run, 0)
        zero_processed_run_months = longest_zero_run_within_window(processed_series, last_pre_time, series_end)
        out = {
            "status": "ok",
            "reason": "right_censored",
            "threshold": threshold,
            "dip_time": dip_time,
            "last_pre_time": last_pre_time,
            "rebound_time": pd.NaT,
            "suppression_duration_months": duration_months,
            "is_right_censored": True,
            "series_end": series_end,
            "pre2020_baseline_start": baseline_start,
            "pre2020_longest_below_threshold_run_months": pre2020_longest_below_threshold_run,
            "pre2020_effective_baseline_run_months": pre2020_effective_baseline_run,
            "excess_gap_months": excess_gap_months,
            "suppression_zero_processed_run_months": zero_processed_run_months,
            "country_level_no_excess_gap_warning": False,
        }
        if zero_processed_run_months >= excess_gap_months and excess_gap_months > 0:
            out["status"] = "excluded"
            out["reason"] = "zero_testing_explains_excess_gap"
        return out

    rebound_time = rebound_candidates.index.min()
    duration_months = month_diff(last_pre_time, rebound_time)
    excess_gap_months = max(duration_months - pre2020_effective_baseline_run, 0)
    zero_processed_run_months = longest_zero_run_within_window(processed_series, last_pre_time, rebound_time)
    out = {
        "status": "ok",
        "reason": "observed_rebound",
        "threshold": threshold,
        "dip_time": dip_time,
        "last_pre_time": last_pre_time,
        "rebound_time": rebound_time,
        "suppression_duration_months": duration_months,
        "is_right_censored": False,
        "series_end": series.index.max(),
        "pre2020_baseline_start": baseline_start,
        "pre2020_longest_below_threshold_run_months": pre2020_longest_below_threshold_run,
        "pre2020_effective_baseline_run_months": pre2020_effective_baseline_run,
        "excess_gap_months": excess_gap_months,
        "suppression_zero_processed_run_months": zero_processed_run_months,
        "country_level_no_excess_gap_warning": False,
    }
    if zero_processed_run_months >= excess_gap_months and excess_gap_months > 0:
        out["status"] = "excluded"
        out["reason"] = "zero_testing_explains_excess_gap"
    return out


def build_results(
    monthly_df: pd.DataFrame,
    anchor_date: pd.Timestamp,
    threshold_divisor: float,
    min_months: int,
    country_dq: pd.DataFrame,
    dq_low_activity_max_monthly: float,
    dq_pre2020_min_max_monthly: float,
    dq_post_min_nonzero_months: int,
    pre2020_monitoring_threshold: float,
    pre2020_max_lookback_years: int,
    pre2020_baseline_gap_cap_months: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    country_dq_idx = country_dq.set_index("country")

    for country, country_df in monthly_df.groupby("COUNTRY_AREA_TERRITORY"):
        country_df = country_df.set_index("month_start").sort_index()
        full_month_idx = pd.date_range(country_df.index.min(), country_df.index.max(), freq="MS")
        country_diag = country_dq_idx.loc[country]
        processed_series = country_df[SPEC_PROCESSED_COLUMN].reindex(full_month_idx, fill_value=0.0)
        # SPEC_PROCESSED_NB is influenza-focused; keep months with any pathogen detections
        # from being treated as "zero testing" in suppression-window checks.
        any_pathogen_positive = sum(
            country_df[spec["column"]].reindex(full_month_idx, fill_value=0.0) for spec in PATHOGEN_SPECS
        ) > 0
        processed_series_for_zero_testing = processed_series.mask(
            (processed_series <= 0) & any_pathogen_positive,
            1.0,
        )
        country_rows: list[dict] = []

        for spec in PATHOGEN_SPECS:
            column = spec["column"]
            key = spec["key"]
            series = country_df[column].reindex(full_month_idx, fill_value=0.0)
            dq_exclude, dq_reason, dq_set_triggered, months_above_low_activity = get_dq_exclusion_reason(
                series=series,
                pathogen_key=key,
                country_diag=country_diag,
                dq_low_activity_max_monthly=dq_low_activity_max_monthly,
                dq_pre2020_min_max_monthly=dq_pre2020_min_max_monthly,
                dq_post_min_nonzero_months=dq_post_min_nonzero_months,
            )

            if dq_exclude:
                out = empty_exclusion(dq_reason, series.index.max())
            else:
                out = evaluate_suppression(
                    series=series,
                    processed_series=processed_series_for_zero_testing,
                    anchor_date=anchor_date,
                    threshold_divisor=threshold_divisor,
                    min_months=min_months,
                    pre2020_monitoring_threshold=pre2020_monitoring_threshold,
                    pre2020_max_lookback_years=pre2020_max_lookback_years,
                    pre2020_baseline_gap_cap_months=pre2020_baseline_gap_cap_months,
                )

            dq_longest_run = np.nan
            if dq_set_triggered in {"set3", "set4", "set6"}:
                dq_longest_run = country_diag.get(f"{dq_set_triggered}_longest_zero_run_months", np.nan)

            country_rows.append(
                {
                    "country": country,
                    "pathogen": key,
                    "pathogen_label": spec["label"],
                    "pathogen_column": column,
                    "max_count": float(series.max()),
                    "n_months": int(series.shape[0]),
                    "dq_pass": bool(not dq_exclude),
                    "dq_reason": dq_reason if dq_exclude else "",
                    "dq_set_triggered": dq_set_triggered,
                    "dq_longest_zero_run_months": dq_longest_run,
                    "months_above_low_activity_threshold": months_above_low_activity,
                    "post_nonzero_months": int(country_diag.get(f"{key}_post_nonzero_months", 0)),
                    **out,
                }
            )

        candidate_rows = [row for row in country_rows if row["dq_pass"] and row["status"] == "ok"]
        if candidate_rows:
            any_candidate_has_gap = any(float(row.get("excess_gap_months", np.nan)) > 0 for row in candidate_rows)
            if not any_candidate_has_gap:
                for row in candidate_rows:
                    row["status"] = "excluded"
                    row["reason"] = "no_excess_gap_vs_pre2020_country_level"
            else:
                for row in candidate_rows:
                    if float(row.get("excess_gap_months", np.nan)) <= 0:
                        row["country_level_no_excess_gap_warning"] = True

        rows.extend(country_rows)
    return pd.DataFrame(rows)


def make_violin_plot(results: pd.DataFrame, output_figure: Path) -> None:
    plot_df = results[results["status"] == "ok"].copy()
    if plot_df.empty:
        raise RuntimeError("No valid suppression durations were found. Plot cannot be generated.")

    labels = [spec["label"] for spec in PATHOGEN_SPECS]
    datasets = {
        label: plot_df.loc[plot_df["pathogen_label"] == label, "suppression_duration_months"].dropna().astype(float).values
        for label in labels
    }

    fig, ax = plt.subplots(figsize=(10, 5.8))
    positions = np.arange(1, len(labels) + 1)

    non_empty_labels = [label for label in labels if datasets[label].size > 0]
    non_empty_positions = [labels.index(label) + 1 for label in non_empty_labels]
    non_empty_data = [datasets[label] for label in non_empty_labels]

    violin = None
    if non_empty_data:
        violin = ax.violinplot(non_empty_data, positions=non_empty_positions, widths=0.85, showmeans=False, showmedians=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    if violin is not None:
        for body, color in zip(violin["bodies"], [colors[labels.index(lbl)] for lbl in non_empty_labels]):
            body.set_facecolor(color)
            body.set_alpha(0.6)
            body.set_edgecolor("black")
            body.set_linewidth(0.8)
        violin["cmedians"].set_color("black")
        violin["cmedians"].set_linewidth(1.2)

    rng = np.random.default_rng(42)
    for pos, label in zip(positions, labels):
        subset = plot_df[plot_df["pathogen_label"] == label]
        x_jitter = rng.normal(loc=pos, scale=0.06, size=subset.shape[0])
        uncensored = subset[~subset["is_right_censored"]]
        censored = subset[subset["is_right_censored"]]

        if not uncensored.empty:
            ax.scatter(
                x_jitter[: uncensored.shape[0]],
                uncensored["suppression_duration_months"],
                s=18,
                alpha=0.6,
                color="black",
                marker="o",
                linewidths=0,
                zorder=3,
            )
        if not censored.empty:
            ax.scatter(
                x_jitter[uncensored.shape[0] :],
                censored["suppression_duration_months"],
                s=28,
                alpha=0.9,
                color="black",
                marker="x",
                linewidths=1,
                zorder=4,
            )

        y_top = subset["suppression_duration_months"].max()
        if pd.isna(y_top):
            y_top = 0.0
        annotation = f"n={subset.shape[0]}\ncens={int(censored.shape[0])}"
        ax.text(pos, y_top + 0.7, annotation, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Suppression duration (months)")
    ax.set_title("COVID-era suppression duration across countries (FluNet)")
    ax.grid(axis="y", alpha=0.2)

    # Legend for censoring markers.
    ax.scatter([], [], marker="o", color="black", s=18, label="Observed rebound")
    ax.scatter([], [], marker="x", color="black", s=28, label="Right-censored")
    ax.legend(frameon=False, loc="upper right")

    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_figure, dpi=300)
    plt.close(fig)


def summarize_by_pathogen(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby("pathogen_label")
        .agg(
            n_country_pathogen=("suppression_duration_months", "size"),
            n_right_censored=("is_right_censored", "sum"),
            median_months=("suppression_duration_months", "median"),
            iqr_low=("suppression_duration_months", lambda x: np.quantile(x, 0.25)),
            iqr_high=("suppression_duration_months", lambda x: np.quantile(x, 0.75)),
            mean_months=("suppression_duration_months", "mean"),
        )
        .reset_index()
    )
    return summary


def sanitize_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = token.strip("._-")
    return token or "unknown"


def export_country_pathogen_timeseries(monthly_df: pd.DataFrame, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0
    used_names: set[str] = set()

    for country, country_df in monthly_df.groupby("COUNTRY_AREA_TERRITORY"):
        country_df = country_df.set_index("month_start").sort_index()
        full_month_idx = pd.date_range(country_df.index.min(), country_df.index.max(), freq="MS")
        processed_series = country_df[SPEC_PROCESSED_COLUMN].reindex(full_month_idx, fill_value=0.0)
        country_token = sanitize_filename_token(str(country))

        for spec in PATHOGEN_SPECS:
            pathogen_series = country_df[spec["column"]].reindex(full_month_idx, fill_value=0.0)
            out_df = pd.DataFrame(
                {
                    "country": country,
                    "pathogen": spec["key"],
                    "pathogen_label": spec["label"],
                    "pathogen_column": spec["column"],
                    "month_start": full_month_idx,
                    "count": pathogen_series.values,
                    "spec_processed_nb": processed_series.values,
                }
            )

            base_name = f"{country_token}__{spec['key']}.csv"
            file_name = base_name
            suffix = 2
            while file_name in used_names:
                file_name = f"{country_token}__{spec['key']}__{suffix}.csv"
                suffix += 1
            used_names.add(file_name)

            out_df.to_csv(output_dir / file_name, index=False)
            files_written += 1

    return files_written


def choose_example_countries(
    results: pd.DataFrame,
    n_countries: int,
    random_seed: int,
    required_pathogen_key: str | None = None,
    require_all_pathogens: bool = True,
) -> list[str]:
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No valid country-pathogen results are available for example-country plotting.")

    if required_pathogen_key is not None:
        ok = ok[ok["pathogen"] == required_pathogen_key]
        if ok.empty:
            raise RuntimeError(
                f"No valid country-pathogen rows found for example-country-pathogen='{required_pathogen_key}'."
            )

    if require_all_pathogens:
        n_required = len(PATHOGEN_SPECS)
        ok_counts = ok.groupby("country")["pathogen"].nunique()
        eligible_countries = set(ok_counts[ok_counts == n_required].index)
        if not eligible_countries:
            raise RuntimeError("No countries found where all six pathogens pass inclusion tests (status == ok).")
        eligible = np.array(sorted(set(ok["country"]).intersection(eligible_countries)))
    else:
        eligible = np.array(sorted(ok["country"].unique()))

    if eligible.size == 0:
        if require_all_pathogens:
            raise RuntimeError(
                "No countries satisfy both all-six-pathogens inclusion and the requested example-country-pathogen filter."
            )
        raise RuntimeError("No countries satisfy the requested example-country-pathogen filter.")

    n_pick = min(max(n_countries, 1), len(eligible))
    rng = np.random.default_rng(random_seed)
    chosen = rng.choice(eligible, size=n_pick, replace=False)
    return sorted(chosen.tolist())


def plot_example_country_timeseries(
    monthly_df: pd.DataFrame,
    results: pd.DataFrame,
    example_countries: list[str],
    output_path: Path,
) -> None:
    monthly_idx = monthly_df.set_index(["COUNTRY_AREA_TERRITORY", "month_start"]).sort_index()
    result_idx = results.set_index(["country", "pathogen"])
    n_rows = len(PATHOGEN_SPECS)
    n_cols = len(example_countries)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(2.8 * n_cols, 12), max(2.0 * n_rows, 10)),
        sharex=True,
        sharey="row",
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, spec in enumerate(PATHOGEN_SPECS):
        pathogen_key = spec["key"]
        pathogen_label = spec["label"]
        pathogen_col = spec["column"]

        for col_idx, country in enumerate(example_countries):
            ax = axes[row_idx, col_idx]

            try:
                country_series = monthly_idx.loc[country, pathogen_col].sort_index()
            except KeyError:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=8)
                ax.set_axis_off()
                continue

            full_idx = pd.date_range(country_series.index.min(), country_series.index.max(), freq="MS")
            country_series = country_series.reindex(full_idx, fill_value=0.0)
            ax.plot(country_series.index, country_series.values, color="#2563eb", linewidth=1)

            if (country, pathogen_key) in result_idx.index:
                info = result_idx.loc[(country, pathogen_key)]
                if isinstance(info, pd.DataFrame):
                    info = info.iloc[0]

                status = info["status"]
                reason = info["reason"]
                dip_time = pd.to_datetime(info["dip_time"], errors="coerce")
                last_pre_time = pd.to_datetime(info["last_pre_time"], errors="coerce")
                rebound_time = pd.to_datetime(info["rebound_time"], errors="coerce")
                duration = info["suppression_duration_months"]
                is_censored = bool(info["is_right_censored"]) if pd.notna(info["is_right_censored"]) else False
                series_end = pd.to_datetime(info["series_end"], errors="coerce")
                no_excess_gap_warning = bool(info.get("country_level_no_excess_gap_warning", False))

                if status == "ok" and pd.notna(last_pre_time):
                    end_time = rebound_time if pd.notna(rebound_time) else series_end
                    if pd.notna(end_time):
                        y_level = float(country_series.max()) * 0.9
                        ax.plot([last_pre_time, end_time], [y_level, y_level], color="black", linewidth=1.2)
                        ax.scatter([last_pre_time], [y_level], color="black", s=10, zorder=3)
                        if pd.notna(rebound_time):
                            ax.scatter([rebound_time], [y_level], color="black", s=10, zorder=3)
                        label = f"{int(duration)} mo" if pd.notna(duration) else "NA"
                        if is_censored:
                            label = f"{int(duration)}+ mo"
                        ax.text(
                            last_pre_time + (end_time - last_pre_time) / 2,
                            y_level,
                            label,
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="black",
                        )
                    if pd.notna(dip_time):
                        ax.axvline(dip_time, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.6)
                    if no_excess_gap_warning:
                        ax.text(
                            0.02,
                            0.95,
                            "no excess gap vs pre2020",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            fontsize=6,
                            color="#b45309",
                        )
                else:
                    ax.text(
                        0.5,
                        0.88,
                        reason.replace("_", " "),
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=7,
                        color="#b91c1c",
                    )

            if row_idx == 0:
                ax.set_title(country, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(pathogen_label, fontsize=9)

            ax.grid(axis="y", alpha=0.15)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        "FluNet monthly time series in random countries with annotated suppression duration",
        fontsize=12,
        y=0.995,
    )
    for ax in axes[-1, :]:
        ax.set_xlabel("Month", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    anchor_date = pd.to_datetime(args.anchor_date)
    if np.isnan(anchor_date.value):
        raise ValueError(f"Invalid anchor date: {args.anchor_date}")
    if args.threshold_divisor <= 0:
        raise ValueError("threshold-divisor must be > 0")
    if args.pre2020_monitoring_threshold < 0:
        raise ValueError("pre2020-monitoring-threshold must be >= 0")
    if args.pre2020_max_lookback_years <= 0:
        raise ValueError("pre2020-max-lookback-years must be > 0")
    if args.pre2020_baseline_gap_cap_months < 0:
        raise ValueError("pre2020-baseline-gap-cap-months must be >= 0")
    if args.dq_zero_run_months < 0:
        raise ValueError("dq-zero-run-months must be >= 0")
    if args.dq_low_activity_max_monthly < 0:
        raise ValueError("dq-low-activity-max-monthly must be >= 0")
    if args.dq_pre2020_min_max_monthly < 0:
        raise ValueError("dq-pre2020-min-max-monthly must be >= 0")
    if args.dq_post_min_nonzero_months < 0:
        raise ValueError("dq-post-min-nonzero-months must be >= 0")

    dq_post_start_date = pd.to_datetime(args.dq_post_start_date)
    if np.isnan(dq_post_start_date.value):
        raise ValueError(f"Invalid dq-post-start-date: {args.dq_post_start_date}")

    monthly_df = load_and_aggregate_monthly(args.input)
    n_timeseries_files = export_country_pathogen_timeseries(
        monthly_df=monthly_df,
        output_dir=args.output_timeseries_dir,
    )
    country_dq = build_country_data_quality_diagnostics(
        monthly_df=monthly_df,
        min_months=args.min_months,
        dq_zero_run_months=args.dq_zero_run_months,
        dq_low_activity_max_monthly=args.dq_low_activity_max_monthly,
        dq_pre2020_min_max_monthly=args.dq_pre2020_min_max_monthly,
        dq_post_start_date=dq_post_start_date,
        dq_post_min_nonzero_months=args.dq_post_min_nonzero_months,
    )

    results = build_results(
        monthly_df=monthly_df,
        anchor_date=anchor_date,
        threshold_divisor=args.threshold_divisor,
        min_months=args.min_months,
        country_dq=country_dq,
        dq_low_activity_max_monthly=args.dq_low_activity_max_monthly,
        dq_pre2020_min_max_monthly=args.dq_pre2020_min_max_monthly,
        dq_post_min_nonzero_months=args.dq_post_min_nonzero_months,
        pre2020_monitoring_threshold=args.pre2020_monitoring_threshold,
        pre2020_max_lookback_years=args.pre2020_max_lookback_years,
        pre2020_baseline_gap_cap_months=args.pre2020_baseline_gap_cap_months,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)

    args.output_dq_csv.parent.mkdir(parents=True, exist_ok=True)
    country_dq.to_csv(args.output_dq_csv, index=False)

    summary = summarize_by_pathogen(results)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)

    make_violin_plot(results, args.output_figure)
    example_countries = choose_example_countries(
        results=results,
        n_countries=args.n_example_countries,
        random_seed=args.random_seed,
        required_pathogen_key=args.example_country_pathogen,
        require_all_pathogens=not args.allow_partial_example_countries,
    )
    plot_example_country_timeseries(
        monthly_df=monthly_df,
        results=results,
        example_countries=example_countries,
        output_path=args.output_example_figure,
    )

    excluded_counts = results[results["status"] != "ok"].groupby("reason").size().to_dict()
    dq_excluded_counts = results[~results["dq_pass"]].groupby("dq_reason").size().to_dict()
    print(f"Saved detailed results: {args.output_csv}")
    print(f"Saved country DQ diagnostics: {args.output_dq_csv}")
    print(f"Saved summary results: {args.output_summary_csv}")
    print(f"Saved country-pathogen timeseries CSVs ({n_timeseries_files}): {args.output_timeseries_dir}")
    print(f"Saved violin plot: {args.output_figure}")
    print(f"Saved example time-series plot: {args.output_example_figure}")
    print(f"Example countries ({len(example_countries)}): {', '.join(example_countries)}")
    print(f"Total country-pathogen pairs: {results.shape[0]}")
    print(f"Included in plot: {(results['status'] == 'ok').sum()}")
    print(f"DQ exclusions by reason: {dq_excluded_counts}")
    print(f"Excluded by reason: {excluded_counts}")


if __name__ == "__main__":
    main()