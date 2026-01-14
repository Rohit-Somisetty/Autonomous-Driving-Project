"""Matplotlib plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update({"figure.autolayout": True})


def plot_overall_event_rates(overall_df: pd.DataFrame, out_path: Path) -> None:
    """Bar plot of event rates per 1,000 miles."""

    if overall_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(overall_df["event_name"], overall_df["rate_per_1k_miles"], color="#1f77b4")
    ax.set_ylabel("Events / 1k miles")
    ax.set_xlabel("Event type")
    ax.set_title("Overall Event Rates")
    for idx, value in enumerate(overall_df["rate_per_1k_miles"]):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_slice_event_rates(
    slice_df: pd.DataFrame, slice_name: str, out_path: Path
) -> None:
    """Grouped bar chart per slice column."""

    subset = slice_df[slice_df["slice_name"] == slice_name]
    if subset.empty:
        return

    slice_values = subset["slice_value"].unique()
    event_names = subset["event_name"].unique()
    x = range(len(slice_values))
    width = 0.8 / max(len(event_names), 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, event_name in enumerate(event_names):
        event_data = subset[subset["event_name"] == event_name]
        value_rates = (
            event_data.set_index("slice_value")["rate_per_1k_miles"].to_dict()
        )
        rates = [value_rates.get(value, 0.0) for value in slice_values]
        positions = [pos + idx * width for pos in x]
        ax.bar(positions, rates, width=width, label=event_name)

    ax.set_xticks([pos + width * (len(event_names) - 1) / 2 for pos in x])
    ax.set_xticklabels(slice_values)
    ax.set_ylabel("Events / 1k miles")
    ax.set_xlabel(slice_name.replace("_", " ").title())
    ax.set_title(f"Event Rates by {slice_name.replace('_', ' ').title()}")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_driver_comparison(driver_metrics: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart comparing baseline vs candidate rates."""

    if driver_metrics.empty or "driver_version" not in driver_metrics:
        return

    pivot = driver_metrics.pivot_table(
        index="event_name",
        columns="driver_version",
        values="rate_per_1k_miles",
        aggfunc="first",
    )
    if pivot.empty or "baseline" not in pivot.columns or "candidate" not in pivot.columns:
        return

    events = pivot.index.tolist()
    x = range(len(events))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([pos - width / 2 for pos in x], pivot["baseline"], width=width, label="Baseline")
    ax.bar([pos + width / 2 for pos in x], pivot["candidate"], width=width, label="Candidate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(events)
    ax.set_ylabel("Events / 1k miles")
    ax.set_title("Driver Version Comparison")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
