# plot.py
# Ve bieu do Forgetting Score - doc so lieu tu CSV
# Chay: python plot.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

FIGURES_DIR = "figures"
RESULTS_DIR = "results"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi"     : 150,
    "font.size"      : 11,
    "axes.titlesize" : 13,
    "axes.labelsize" : 11,
    "legend.fontsize": 10,
})


# --- DOC SO LIEU TU CSV ---
def load_csv():
    cmp     = pd.read_csv(f"{RESULTS_DIR}/comparison_3methods.csv")
    tasks   = cmp["Task"].tolist()
    f_seq   = dict(zip(tasks, cmp["Sequential"]))
    f_ewc   = dict(zip(tasks, cmp["EWC"]))
    f_qlora = dict(zip(tasks, cmp["QLoRA"]))
    return f_seq, f_ewc, f_qlora


# --- HINH: SO SANH FORGETTING SCORE (BAR CHART) ---
def plot_forgetting_bar(f_seq, f_ewc, f_qlora):
    tasks = ["HellaSwag", "MedMCQA"]
    seq   = [f_seq["Forgetting HellaSwag (%)"],   f_seq["Forgetting MedMCQA (%)"]]
    ewc   = [f_ewc["Forgetting HellaSwag (%)"],   f_ewc["Forgetting MedMCQA (%)"]]
    qlora = [f_qlora["Forgetting HellaSwag (%)"], f_qlora["Forgetting MedMCQA (%)"]]

    x, w  = np.arange(2), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    for vals, offset, label, hatch in [
        (seq,   -w, "Sequential",          "///"),
        (ewc,    0, "EWC (λ=5000)",        "..."),
        (qlora,  w, "QLoRA (r=16, 4-bit)", "xxx"),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label,
                      color="white", edgecolor="black",
                      linewidth=1.5, hatch=hatch)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + w/2, h + 0.15, f"{h}%",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12)
    ax.set_ylabel("Forgetting Score (%)")
    ax.set_title("So sánh Forgetting Score: Sequential vs EWC vs QLoRA\n"
                 "(Thấp hơn = ít quên = tốt hơn)")
    ax.set_ylim(0, 14)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/compare_forgetting_3methods.png", facecolor="white")
    plt.close()
    print("✅ compare_forgetting_3methods.png")


# --- MAIN ---
if __name__ == "__main__":
    print("=" * 50)
    print("  VE BIEU DO FORGETTING SCORE")
    print("=" * 50)
    f_seq, f_ewc, f_qlora = load_csv()
    plot_forgetting_bar(f_seq, f_ewc, f_qlora)
    print("\n✅ Xong! Kiem tra thu muc figures/")