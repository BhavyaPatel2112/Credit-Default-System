"""M2 figures for the EDA chapter."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import PROC, ROOT

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
e = json.load(open(ROOT / "reports" / "eda.json"))
BLUE, GREY = "#2a6fdb", "#9aa5b1"


def bars(ax, d, order, title):
    vals = [d[k]["default_rate"] * 100 for k in order]
    ax.bar(order, vals, color=BLUE)
    ax.axhline(e["base_rate"] * 100, ls="--", color=GREY, lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_ylabel("Default rate (%)"); ax.spines[["top", "right"]].set_visible(False)


fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), dpi=160, gridspec_kw={"width_ratios": [1, 2]})
bars(axs[0], e["default_rate_by_group"]["gender"], ["Female", "Male"], "By gender")
bars(axs[1], e["default_rate_by_group"]["age_band"], ["<25", "25-34", "35-44", "45-54", "55+"], "By age band")
fig.suptitle("Default rate differs by group before any model is trained (dashed line = 8.07% overall)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "eda_default_by_group.png")

df = pd.read_parquet(PROC / "applicants.parquet", columns=["TARGET", "DAYS_EMPLOYED_ANOM", "EXT_SOURCE_1",
                                                           "EXT_SOURCE_2", "EXT_SOURCE_3"])
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), dpi=160)
for ax, c in zip(axs, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
    for t, col, lab in [(0, GREY, "Repaid"), (1, BLUE, "Defaulted")]:
        ax.hist(df.loc[df.TARGET == t, c].dropna(), bins=40, density=True, alpha=0.6, color=col, label=lab)
    ax.set_title(f"{c} ({df[c].isna().mean():.0%} missing)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("External credit scores separate defaulters well, but EXT_SOURCE_1 is missing for over half", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "eda_ext_sources.png")
print("figures written")
