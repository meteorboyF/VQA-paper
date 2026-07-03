"""E8d - Regenerate F8 with the full method battery (CPU, seconds).

The current F8 only shows the two E7b methods, while Table 3 in the paper
covers the E7c methods as well. This rebuilds F8 from the E7b + E7c summaries
(one panel), and adds a second BLIP panel automatically once E7d has run -
turning the figure into visual evidence that the negative result holds
across two frozen VQA models.

No DONE marker: rerunning is free and should always pick up the newest
summaries.
"""
import os

from src import config, env, figures, progress, resultlog

EXP = "E8D"
RESULTS_E8D = os.path.join(config.RESULTS, "E8d_figures")


def main():
    progress.install_error_hook("E8d F8 regeneration")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E8D, exist_ok=True)

    e7b_summary = os.path.join(config.RESULTS, "E7b_diagnostics", "summary.json")
    e7c_summary = os.path.join(config.RESULTS, "E7c_risk_signals", "summary.json")
    e7d_summary = os.path.join(config.RESULTS, "E7d_blip_diagnostics", "summary.json")
    missing = [p for p in (e7b_summary, e7c_summary) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Run E7b/E7c first; missing: {missing}")

    figures.set_fig_dir(config.FIGURES_DIR)
    has_blip = os.path.exists(e7d_summary)
    print(f"[E8d] Rebuilding F8 (BLIP panel: {'yes' if has_blip else 'not yet - run E6b+E7d'})")
    fig_path = figures.f8_selective_diagnostics_full(
        e7b_summary, e7c_summary, e7d_summary if has_blip else None)

    resultlog.log_run(EXP,
                      metrics={"figure": fig_path, "blip_panel": has_blip},
                      params={"backbones": config.BACKBONES},
                      results_dir=RESULTS_E8D, repo_root=config.REPO_ROOT)
    print(f"[E8d DONE] {fig_path} - copy into 'IEEE Access template/figures/' "
          f"to update the manuscript.")
