#%%
"""Log analysis for Dotter session JSONL (matplotlib only)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_data(filename: str | Path) -> list[dict]:
    """Load and parse log file data (one JSON object per line)."""
    objects = []
    path = Path(filename)
    with open(path, "r") as f:
        for line in f:
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return objects


def get_user_counts(objects: list[dict]) -> dict[str, int]:
    """Count entries per username and print sorted results with total time."""
    username_counts: dict[str, int] = {}
    username_times: dict[str, float] = {}
    for data in objects:
        username = data.get("username", "unknown").strip()
        username_counts[username] = username_counts.get(username, 0) + 1
        username_times[username] = username_times.get(username, 0) + data.get("time_elapsed", 0)

    rows = sorted(
        [
            (u, username_counts[u], username_times[u])
            for u in username_counts
        ],
        key=lambda r: r[1],
        reverse=True,
    )
    print("\nEntries and time per username:")
    print(f"{'Username':<20} {'Entries':>8} {'Total Time (s)':>16}")
    for u, c, t in rows:
        print(f"{u:<20} {c:>8} {t:>16.2f}")
    return username_counts


def get_user_entries(objects: list[dict], username: str) -> list[dict]:
    u = username.strip()
    return [
        d
        for d in objects
        if d.get("username", "").strip() in (u, "null")
        or (u == "null" and not d.get("username"))
    ]


def calculate_metrics(entries: list[dict]):
    """WPM, scan times, delay STDs, outlier rates, cumulative time (normalized)."""
    wpm_values = [
        len(entry["best_val"][:-1]) / entry["time_elapsed"] * 60 / 5 for entry in entries
    ]

    scan_times = [
        entry["time_elapsed"] / (len(entry["delay_pairs"]) - 1) - entry["delay_pairs"][0]["period"] / 2
        for entry in entries
    ]

    outlier_rates = []
    for entry in entries:
        delays = [pair["delay"] for pair in entry["delay_pairs"]]
        outliers = [d for d in delays if abs(d) > entry["delay_pairs"][0]["period"] * 0.25]
        outlier_rate = 2 * len(outliers) / len(delays) if delays else 0
        outlier_rates.append(outlier_rate)

    delay_stds = [
        np.std([pair["delay"] for pair in entry["delay_pairs"] if abs(pair["delay"]) < 0.4])
        for entry in entries
    ]

    cum_time_elapsed = np.array(
        [sum(entry["time_elapsed"] for entry in entries[: i + 1]) for i in range(len(entries))]
    )
    if len(cum_time_elapsed) and cum_time_elapsed[-1] > 0:
        cum_time_elapsed *= 7200 / cum_time_elapsed[-1]

    return wpm_values, scan_times, delay_stds, outlier_rates, cum_time_elapsed


def calculate_rolling_average(values: list[float], alpha: float = 0.05) -> list[float]:
    rolling_avg = [values[0]]
    ema = values[0]
    for val in values[1:]:
        ema = alpha * val + (1 - alpha) * ema
        rolling_avg.append(ema)
    return rolling_avg


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _user_colors(n: int) -> list:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if n <= len(cycle):
        return [cycle[i] for i in range(n)]
    return [plt.cm.hsv(i / n) for i in range(n)]


def plot_metric(ax, x, y, rolling_avg, title, ylabel, username: str) -> None:
    t_min = x / 60.0
    ax.scatter(t_min, y, alpha=0.3, label=f"{username} Raw", s=12)
    ax.plot(t_min, rolling_avg, color="red", linewidth=2, label=f"{username} Rolling Avg")
    if len(t_min) >= 2:
        slope, intercept, _, _, _ = stats.linregress(t_min, y)
        ax.plot(t_min, slope * t_min + intercept, "g--", alpha=0.8, label="Linear fit")
    ax.set_xlabel("Time Elapsed (minutes)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=15)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    _despine(ax)


def plot_user_metrics(entries: list[dict], username: str) -> None:
    wpm_values, scan_times, delay_stds, outlier_rates, cum_time_elapsed = calculate_metrics(entries)
    plt.rcParams["figure.figsize"] = [15, 10]
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(f'Performance Metrics for User "{username}"', fontsize=16, y=1.02)

    metrics = [
        (wpm_values, "WPM vs Time", "Words Per Minute"),
        (scan_times, "Avg Scan Time vs Time", "Avg Scan Time (seconds)"),
        (delay_stds, "Delay Standard Deviation vs Time", "Std Dev of Delays (seconds)"),
        (outlier_rates, "Outlier Rate vs Time", "Outlier Rate"),
    ]

    for (metric, title, ylabel), ax in zip(metrics, axes.flat):
        rolling_avg = calculate_rolling_average(metric)
        plot_metric(ax, cum_time_elapsed, metric, rolling_avg, title, ylabel, username)

    plt.tight_layout()
    for ax in axes.flat:
        _despine(ax)
    plt.show()


def plot_comparative_metrics(objects: list[dict], usernames: list[str]) -> None:
    plt.rcParams["figure.figsize"] = [15, 10]
    fig, axes = plt.subplots(2, 2)
    fig.suptitle("Comparative Performance Metrics", fontsize=16, y=1.02)

    metric_names = [
        "WPM vs Time",
        "Avg Scan Time vs Time",
        "Delay Standard Deviation vs Time",
        "Outlier Rate vs Time",
    ]
    ylabels = [
        "Words Per Minute",
        "Avg Scan Time (seconds)",
        "Std Dev of Delays (seconds)",
        "Outlier Rate",
    ]

    all_rolling_avgs: dict[int, list] = {i: [] for i in range(4)}
    all_times: list[np.ndarray] = []
    colors = _user_colors(len(usernames))

    for idx, username in enumerate(usernames):
        entries = get_user_entries(objects, username)
        metrics = calculate_metrics(entries)
        all_times.append(metrics[4])

        for i, (metric, ax) in enumerate(zip(metrics[:4], axes.flat)):
            rolling_avg = calculate_rolling_average(metric)
            t_min = metrics[4] / 60.0
            ax.scatter(t_min, metric, alpha=0.3, color=colors[idx], label=username, s=10)
            ax.plot(t_min, rolling_avg, alpha=0.8, color=colors[idx])
            all_rolling_avgs[i].append(rolling_avg)
            ax.set_title(metric_names[i], pad=15)
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel(ylabels[i])

    min_len = min(len(t) for t in all_times)
    common_time = np.linspace(0, min_len, min_len, dtype=float)

    for i, ax in enumerate(axes.flat):
        interpolated_avgs = []
        for j, ra in enumerate(all_rolling_avgs[i]):
            interp_func = np.interp(common_time, np.arange(len(ra)), ra)
            interpolated_avgs.append(interp_func)
        mean_rolling_avg = np.mean(interpolated_avgs, axis=0)
        x_mean = np.arange(len(mean_rolling_avg)) * 120 / max(len(mean_rolling_avg), 1)
        ax.plot(x_mean, mean_rolling_avg, color="black", linewidth=3, label="Group Average")
        ylims = [(0, 10), (0, 2.4), (0, 0.25), (0, 0.125)]
        ax.set_ylim(*ylims[i])

    plt.tight_layout()
    for ax in axes.flat:
        _despine(ax)
    plt.show()


def plot_comparative_wpm(objects: list[dict], usernames: list[str]) -> None:
    plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")
    colors = _user_colors(len(usernames))
    all_rolling_avgs = []
    all_times: list[np.ndarray] = []

    for i, username in enumerate(usernames):
        entries = get_user_entries(objects, username)
        wpm_values, _, _, _, cum_time_elapsed = calculate_metrics(entries)
        if len(cum_time_elapsed) == 0:
            continue
        cum = cum_time_elapsed * 120 / cum_time_elapsed[-1]
        ax.scatter(cum, wpm_values, alpha=0.2, color=colors[i], label=f"{username} (Raw)", s=50)
        rolling_avg = calculate_rolling_average(wpm_values)
        ax.plot(cum, rolling_avg, alpha=0.8, color=colors[i], label=f"{username} (EMA)", linewidth=2)
        all_rolling_avgs.append(rolling_avg)
        all_times.append(cum)

    if not all_times:
        plt.close()
        return

    min_len = min(len(t) for t in all_times)
    common_time = np.linspace(0, 120, min_len)
    interpolated_avgs = []
    for i, ra in enumerate(all_rolling_avgs):
        interp_func = np.interp(common_time, all_times[i], ra)
        interpolated_avgs.append(interp_func)
    mean_rolling_avg = np.mean(interpolated_avgs, axis=0)
    ax.plot(common_time, mean_rolling_avg, color="black", linewidth=3, label="Group Average")

    log_x = np.log(common_time + 1)
    log_y = np.log(np.maximum(mean_rolling_avg, 1e-6))
    coeffs = np.polyfit(log_x, log_y, 1)
    a, b = np.exp(coeffs[1]), coeffs[0]
    power_law = a * (common_time + 1) ** b
    ax.plot(common_time, power_law, color="black", linestyle="--", linewidth=2.5, label=f"Power law (y = {a:.2f}(x+1)^{b:.2f})")

    ax.set_ylabel("Words Per Minute (WPM)", fontsize=14, labelpad=12)
    ax.set_xlabel("Time (minutes)", fontsize=14, labelpad=12)
    ax.set_title("Learning curve (WPM)", pad=15, fontsize=16)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="lightgray", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.2, linestyle="-", color="black")
    ax.set_ylim(0, 10)
    _despine(ax)
    plt.tight_layout()
    plt.show()


#%%
if __name__ == "__main__":
    _default_log = Path(__file__).resolve().parent / "log.txt"
    objects = load_data(_default_log)
    get_user_counts(objects)

    target_username = "p4"
    entries = get_user_entries(objects, target_username)
    if entries:
        plot_user_metrics(entries, target_username)

    usernames = ["p1", "P2", "P3", "p4"]
    plot_comparative_wpm(objects, usernames)
