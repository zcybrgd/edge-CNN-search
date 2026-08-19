"""
Hyperparameter sweep over distillation temperature and alpha.

Trains the SAME searched architecture, from the SAME teacher, for a
REDUCED epoch budget per combination (sweep_epochs, e.g. 30-40) -- full
100-epoch runs per combination would be prohibitively slow (a 3x3 grid
at 100 epochs each is effectively 900 epochs of training). Reduced-epoch
runs won't give you final converged accuracy per combination, but they
reliably rank combinations relative to each other, which is all a sweep
needs to tell you: "is my original (T=4.0, alpha=0.7) actually a good
choice, or would something else clearly do better."

This is the same "cheap proxy, ranking not absolute accuracy" logic as
evolutionary_search.py's proxy training -- same tradeoff, same caveat,
applied to hyperparameters instead of architectures.

After the sweep, take the winning (temperature, alpha) pair and rerun
distillation.py's full 100-epoch schedule with those values set in
config.yaml, to get the real final number -- don't report the sweep's
own reduced-epoch accuracy as your headline result.
"""
import argparse
import itertools
import json
import os
import yaml
import torch
import torch.nn.functional as F

from data import get_dataloaders
from teacher_model import build_teacher, evaluate
from student_model import build_student
from search_space import ArchConfig
from utils import setup_logging, set_seed, load_checkpoint, count_params, append_json_log

log = setup_logging(__name__)

TEMPERATURE_GRID = [2.0, 4.0, 8.0]
ALPHA_GRID = [0.5, 0.7, 0.9]
SWEEP_LOG_PATH = "results/sweep_log.json"


def distillation_loss(student_logits, teacher_logits, labels, temperature: float, alpha: float):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def run_one_combo(cfg: dict, arch: ArchConfig, teacher, train_loader, test_loader,
                   device, temperature: float, alpha: float, sweep_epochs: int) -> dict:
    set_seed(cfg["search"]["seed"])  # same seed every combo -> differences are due to (T, alpha) only
    student = build_student(arch, cfg["data"]["num_classes"]).to(device)

    dcfg = cfg["distillation"]
    optimizer = torch.optim.SGD(
        student.parameters(), lr=float(dcfg["lr"]), momentum=0.9,
        weight_decay=float(dcfg["weight_decay"]), nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sweep_epochs)

    best_acc = 0.0
    curve = []
    for epoch in range(sweep_epochs):
        student.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        acc = evaluate(student, test_loader, device)
        curve.append({"epoch": epoch, "loss": total_loss / len(train_loader), "test_acc": acc})
        best_acc = max(best_acc, acc)

    return {"temperature": temperature, "alpha": alpha, "best_acc": best_acc,
            "final_acc": curve[-1]["test_acc"], "curve": curve}


def run_sweep(cfg: dict, sweep_epochs: int = 35):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_dataloaders(cfg)

    teacher = build_teacher(cfg["data"]["num_classes"]).to(device)
    load_checkpoint(teacher, cfg["teacher"]["checkpoint"], device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    with open(cfg["search"]["best_arch_path"]) as f:
        arch = ArchConfig.from_dict(json.load(f))

    log.info("Sweeping %d temperature x %d alpha combos, %d epochs each (%d total epochs of compute)",
              len(TEMPERATURE_GRID), len(ALPHA_GRID), sweep_epochs,
              len(TEMPERATURE_GRID) * len(ALPHA_GRID) * sweep_epochs)

    results = []
    combos = list(itertools.product(TEMPERATURE_GRID, ALPHA_GRID))
    for i, (temperature, alpha) in enumerate(combos):
        log.info("[%d/%d] Running T=%.1f, alpha=%.2f ...", i + 1, len(combos), temperature, alpha)
        result = run_one_combo(cfg, arch, teacher, train_loader, test_loader,
                                device, temperature, alpha, sweep_epochs)
        log.info("[%d/%d] T=%.1f alpha=%.2f -> best_acc=%.4f (final_acc=%.4f)",
                  i + 1, len(combos), temperature, alpha, result["best_acc"], result["final_acc"])
        results.append(result)
        append_json_log(SWEEP_LOG_PATH, {
            "temperature": temperature, "alpha": alpha,
            "best_acc": result["best_acc"], "final_acc": result["final_acc"],
        })

    best = max(results, key=lambda r: r["best_acc"])
    log.info("--- Sweep complete ---")
    log.info("Best combo: T=%.1f, alpha=%.2f -> best_acc=%.4f (at %d epochs, NOT final)",
              best["temperature"], best["alpha"], best["best_acc"], sweep_epochs)
    log.info("Your original default (T=4.0, alpha=0.7) result for comparison:")
    default = next((r for r in results if r["temperature"] == 4.0 and r["alpha"] == 0.7), None)
    if default:
        log.info("  T=4.0, alpha=0.7 -> best_acc=%.4f", default["best_acc"])
    log.info("Next step: set the winning (temperature, alpha) in config.yaml's "
              "distillation section and rerun distillation.py's full schedule "
              "for the real, converged number -- do not report this sweep's "
              "%d-epoch accuracy as final.", sweep_epochs)

    os.makedirs("results", exist_ok=True)
    with open("results/sweep_summary.json", "w") as f:
        json.dump({
            "sweep_epochs": sweep_epochs,
            "results": [{"temperature": r["temperature"], "alpha": r["alpha"],
                         "best_acc": r["best_acc"], "final_acc": r["final_acc"]} for r in results],
            "best": {"temperature": best["temperature"], "alpha": best["alpha"], "best_acc": best["best_acc"]},
        }, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sweep_epochs", type=int, default=35,
                         help="Epoch budget per (temperature, alpha) combo -- reduced vs full "
                              "distillation.py schedule, for ranking not final accuracy.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_sweep(cfg, sweep_epochs=args.sweep_epochs)