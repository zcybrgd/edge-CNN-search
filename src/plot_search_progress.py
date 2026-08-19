import json
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

LOG_PATH = "results/search_log.json"
OUT_PATH = "results/search_progress.png"


def load_log(path: str) -> list:
    if not os.path.exists(path):
        log.error("Search log not found at %s. Run evolutionary_search.py first.", path)
        sys.exit(1)
    with open(path) as f:
        entries = json.load(f)
    if not entries:
        log.error("Search log at %s is empty.", path)
        sys.exit(1)
    return entries


def plot(entries: list, out_path: str):
    # Preserve the order entries were logged in: init population first
    # (all generation == 0), then each generation's single child in order.
    # This "search order" x-axis is what lets us see improvement over time,
    # since generation number alone repeats for the whole init population.
    fitness = np.array([e["fitness"] for e in entries])
    params = np.array([e["params"] for e in entries]) / 1e6   # -> millions
    flops = np.array([e["flops"] for e in entries]) / 1e6     # -> millions
    generations = np.array([e["generation"] for e in entries])

    order = np.arange(len(entries))
    running_best = np.maximum.accumulate(fitness)

    n_init = int((generations == 0).sum())
    log.info("Loaded %d entries (%d initial population + %d generations)",
              len(entries), n_init, len(entries) - n_init)
    log.info("Best fitness found: %.4f at search-order index %d",
              fitness.max(), int(fitness.argmax()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(order[:n_init], fitness[:n_init], color="tab:gray", s=30,
               label="Initial population", zorder=3)
    ax.scatter(order[n_init:], fitness[n_init:], color="tab:blue", s=30,
               label="Evolved children", zorder=3)
    ax.plot(order, running_best, color="tab:red", linewidth=1.5,
            label="Best so far", zorder=2)
    ax.axvline(n_init - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Search order (init population, then each generation's child)")
    ax.set_ylabel("Proxy test accuracy")
    ax.set_title("Search fitness over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    sc = ax.scatter(flops, params, c=fitness, cmap="viridis", s=50, edgecolor="k", linewidth=0.3)
    best_idx = int(fitness.argmax())
    ax.scatter(flops[best_idx], params[best_idx], facecolor="none",edgecolor="red", s=200, linewidth=2, label="Best found")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Proxy test accuracy")
    ax.set_xlabel("FLOPs (M)")
    ax.set_ylabel("Params (M)")
    ax.set_title("Efficiency frontier explored")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info("Saved plot to %s", out_path)
    print("\n--- Search summary ---")
    print(f"Total candidates evaluated: {len(entries)}")
    print(f"Initial population best:   {fitness[:n_init].max():.4f}")
    print(f"Final best (overall):      {fitness.max():.4f}")
    print(f"Improvement over init best: {fitness.max() - fitness[:n_init].max():+.4f}")
    last_gen = generations.max()
    if last_gen >= 5:
        recent = fitness[generations >= last_gen - 4]
        print(f"Best fitness in last 5 generations: {recent.max():.4f} "
        f"(check search_progress.png to see if this has plateaued)")


if __name__ == "__main__":
    entries = load_log(LOG_PATH)
    plot(entries, OUT_PATH)


"""
Produces:
  1. Fitness per candidate over "search order" (init pop first, then each
     generation's child), plus a running best-so-far line shows whether
     the search actually improved over random init or plateaued/regressed.
  2. Params vs FLOPs scatter, colored by fitness shows whether the search
     is finding a useful accuracy/efficiency frontier or just wandering.
"""