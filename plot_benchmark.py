"""Make the benchmark table and chart from results/benchmark.csv.

    python plot_benchmark.py

Writes results/benchmark.md and results/benchmark.png.
"""

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"sklearn": "#2a78d6", "spark": "#eb6834"}
NAMES = {"sklearn": "scikit-learn", "spark": "Spark MLlib"}


def short_number(n, _position=None):
    """10000 -> 10k, 2000000 -> 2M. Used for the axis labels."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n / 1_000:g}k"
    return f"{n:g}"


def write_table(results):
    table = results[["engine", "size", "status", "train_rows", "seconds", "memory_gb", "macro_f1"]]
    # show the missing numbers of failed runs as empty cells
    table = table.astype(str).replace({"nan": "", "<NA>": ""})
    with open("results/benchmark.md", "w") as f:
        f.write(table.to_markdown(index=False) + "\n")
    print(table.to_string(index=False))


def draw_chart(results):
    finished = results[results["status"] == "ok"]
    fig, (time_ax, memory_ax) = plt.subplots(1, 2, figsize=(11, 4.2))

    for engine, runs in finished.groupby("engine"):
        runs = runs.sort_values("train_rows")
        for ax, column in [(time_ax, "seconds"), (memory_ax, "memory_gb")]:
            ax.plot(runs["train_rows"], runs[column], color=COLORS[engine],
                    linewidth=2, marker="o", markersize=7, label=NAMES[engine])
            # write the engine's name next to its last point
            last = runs.iloc[-1]
            ax.annotate(NAMES[engine], (last["train_rows"], last[column]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)

    time_ax.set_title("Time to load, train and predict (seconds)", loc="left")
    time_ax.set_yscale("log")
    time_ax.yaxis.set_major_formatter("{x:g}")  # 1, 10, 100 instead of 10^0, 10^1...
    memory_ax.set_title("Peak memory (GB)", loc="left")
    for ax in (time_ax, memory_ax):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(short_number)
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("Training reviews")
        ax.grid(True, color="#e6e6e6")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    # List the runs that did not finish under the chart
    failed = results[results["status"] != "ok"]
    if len(failed) > 0:
        notes = [f"{NAMES[r.engine]} on {r.size} reviews: {r.status}" for r in failed.itertuples()]
        fig.text(0.01, 0.01, "Did not finish: " + "; ".join(notes), fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig("results/benchmark.png", dpi=150)


if __name__ == "__main__":
    results = pd.read_csv("results/benchmark.csv")
    # keep row counts as whole numbers even when a failed run left them empty
    results["train_rows"] = results["train_rows"].astype("Int64")
    write_table(results)
    draw_chart(results)
