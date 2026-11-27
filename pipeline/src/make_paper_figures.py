"""
Figure 1 of the paper, generated from outputs only.

(a) distribution of T_rec by regime, (b) Kaplan-Meier curves, (c) permutation importance of
the Experiment A default-setting XGBoost LOOCV model (a model that generalises).
Writes figure1.svg (text as outlines), figure1.pdf and figure1.png (300 dpi).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
NAMES = {"global_tourism_idx": "Global tourism index", "Economic_Condition_Index": "ECI", "gdp_growth": "GDP growth",
         "DSI": "DSI", "Season_cos": "Season (cos)", "Season_sin": "Season (sin)", "TWDI": "TWDI",
         "external_shock_flag": "Shock flag"}


def make_figure1(results_dir: Path, figures_dir: Path) -> None:
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 8, "svg.fonttype": "path"})
    ev = pd.read_csv(results_dir / "supplementary_event_table.csv")
    ex = json.loads((results_dir / "paper_extras.json").read_text())
    perm = pd.read_csv(results_dir / "permutation_importance_ExpA_XGBoost.csv")
    crisis = ev.external_shock_flag == 1
    iso, cr = ev[~crisis], ev[crisis]
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 1.55), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    b = np.arange(13)
    ax[0].bar(b - 0.2, [(iso.recovery_months == k).sum() for k in b], 0.4, color="#2b7bba", label=f"Outside 2019–2022 (n = {len(iso)})")
    ax[0].bar(b + 0.2, [(cr.recovery_months == k).sum() for k in b], 0.4, color="#d62728", label=f"2019–2022 (n = {len(cr)})")
    ax[0].set_xlabel("$T_{rec}$ (months)"); ax[0].set_ylabel("Events"); ax[0].set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax[0].legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.12, 1.02)); ax[0].set_title("(a)", loc="left", fontsize=8)
    for key, col, lab in (("outside_crisis", "#2b7bba", "Outside 2019–2022"), ("crisis_window", "#d62728", "2019–2022")):
        x, y = ex["km_curves"][key]
        ax[1].step(x, y, where="post", color=col, label=lab)
    ax[1].set_xlabel("Months since onset"); ax[1].set_ylabel("Proportion not recovered"); ax[1].set_ylim(0, 1.02)
    ax[1].set_xticks([0, 2, 4, 6, 8, 10, 12]); ax[1].legend(frameon=False, fontsize=6.5, loc="center right"); ax[1].set_title("(b)", loc="left", fontsize=8)
    s = perm.sort_values("mean_mae_increase")
    ax[2].barh([NAMES.get(f, f) for f in s.Feature], s.mean_mae_increase, color="#555555")
    ax[2].axvline(0, color="black", lw=0.5)
    ax[2].set_xlabel("MAE increase (months)"); ax[2].set_title("(c)", loc="left", fontsize=8); ax[2].tick_params(axis="y", labelsize=7)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(w_pad=0.8)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(figures_dir / f"figure1.{ext}", **kw)
    plt.close(fig)


if __name__ == "__main__":
    make_figure1(ROOT / "outputs" / "results", ROOT / "outputs" / "figures")
