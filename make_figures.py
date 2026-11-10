"""Run the bed-demand analysis and save its figures to figures/.

Usage: python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from analysis import PROJECT_ROOT, build_figures, run_analysis, summarise_demand

FIGURES_DIR = PROJECT_ROOT / "figures"


def main() -> None:
    results = run_analysis()
    FIGURES_DIR.mkdir(exist_ok=True)
    for name, fig in build_figures(results).items():
        fig.savefig(FIGURES_DIR / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figures/{name}")
    print(summarise_demand(results).to_string())


if __name__ == "__main__":
    main()
