import argparse
import json
import yaml
import torch
import torch.nn as nn
from data import get_dataloaders
from student_model import build_student
from search_space import ArchConfig
from teacher_model import evaluate
from utils import setup_logging, set_seed, save_checkpoint, count_params, append_json_log

log = setup_logging(__name__)


def train_control(cfg: dict):
    set_seed(cfg["search"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_dataloaders(cfg)
    with open(cfg["search"]["best_arch_path"]) as f:
        arch = ArchConfig.from_dict(json.load(f))
    model = build_student(arch, cfg["data"]["num_classes"]).to(device)
    log.info("Control student params: %.2fM (same architecture as distillation.py)",count_params(model) / 1e6)
    dcfg = cfg["distillation"]
    optimizer = torch.optim.SGD(model.parameters(), lr=float(dcfg["lr"]), momentum=0.9,weight_decay=float(dcfg["weight_decay"]), nesterov=True,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dcfg["epochs"])
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = dcfg["checkpoint"].replace("student_distilled.pt", "student_control_no_distill.pt")
    curve_path = "results/control_curve.json"
    best_acc = 0.0
    for epoch in range(dcfg["epochs"]):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        avg_loss = total_loss / len(train_loader)
        log.info("Epoch %d/%d | loss=%.4f | test_acc=%.4f", epoch + 1, dcfg["epochs"], avg_loss, acc)
        append_json_log(curve_path, {"epoch": epoch, "loss": avg_loss, "test_acc": acc})
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, checkpoint_path, extra={"acc": acc, "epoch": epoch})
    log.info("Best control (no-distillation) accuracy: %.4f", best_acc)
    return best_acc


def compare_to_distillation(cfg: dict, control_acc: float):
    import os
    dpath = cfg["distillation"]["checkpoint"]
    if not os.path.exists(dpath):
        log.info("Distilled checkpoint not found yet at %s -- run distillation.py " "and re-run this comparison later.", dpath)
        return
    payload = torch.load(dpath, map_location="cpu")
    distilled_acc = payload.get("acc")
    if distilled_acc is None:
        log.warning("Distilled checkpoint has no 'acc' field, cannot compare.")
        return
    delta = distilled_acc - control_acc
    log.info("--- Distillation effect ---")
    log.info("No-distillation control accuracy: %.4f", control_acc)
    log.info("Distilled student accuracy:  %.4f", distilled_acc)
    log.info("Delta from distillation: %+.4f", delta)
    with open("results/distillation_effect.json", "w") as f:
        json.dump({"control_acc": control_acc,"distilled_acc": distilled_acc, "delta": delta,}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    control_acc = train_control(cfg)
    compare_to_distillation(cfg, control_acc)


"""
Trains the SAME searched architecture (from search/best_arch_path) using
plain supervised learning on hard labels only -- no teacher, no soft
targets. This is the baseline needed to actually claim distillation
helped: without this number, "the distilled student got 61% accuracy"
is uninterpretable -- you don't know if the architecture would have
gotten 61% on its own, or 40%, or 65%.

Same optimizer, schedule, and epoch budget as distillation.py's config
(cfg["distillation"]) is used here too, so the only variable that
differs between this run and distillation.py is the loss function
(hard-label cross-entropy only, vs KD loss). Keeping everything else
identical is what makes the comparison valid -- if the control used a
different LR schedule or epoch count, any accuracy gap could be
attributed to that instead of distillation itself.
"""