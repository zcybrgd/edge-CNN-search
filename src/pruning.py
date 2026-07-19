import argparse
import json
import yaml
import torch
import torch.nn as nn
import torch_pruning as tp
from data import get_dataloaders
from teacher_model import evaluate
from student_model import build_student
from search_space import ArchConfig
from utils import setup_logging, set_seed, save_checkpoint, load_checkpoint, count_params, count_flops
log = setup_logging(__name__)

def build_pruner(model: nn.Module, example_inputs: torch.Tensor, prune_ratio: float):
    importance = tp.importance.MagnitudeImportance(p=1)  # L1-norm importance
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            ignored_layers.append(m)  # never prune the final classifier layer
    pruner = tp.pruner.MagnitudePruner(model, example_inputs,importance=importance,pruning_ratio=prune_ratio,ignored_layers=ignored_layers,)
    return pruner


def finetune(model: nn.Module, cfg: dict, train_loader, test_loader, device, epochs: int, lr: float) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        log.info("  finetune epoch %d/%d | test_acc=%.4f", epoch + 1, epochs, acc)
        best_acc = max(best_acc, acc)
    return best_acc


def run_pruning(cfg: dict):
    set_seed(cfg["search"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_dataloaders(cfg)
    with open(cfg["search"]["best_arch_path"]) as f:
        arch = ArchConfig.from_dict(json.load(f))
    model = build_student(arch, cfg["data"]["num_classes"]).to(device)
    load_checkpoint(model, cfg["distillation"]["checkpoint"], device=device)
    input_shape = (1, 3, cfg["data"]["input_size"], cfg["data"]["input_size"])
    example_inputs = torch.randn(*input_shape).to(device)
    pre_params = count_params(model)
    pre_flops = count_flops(model, input_shape)
    pre_acc = evaluate(model, test_loader, device)
    log.info("Before pruning: params=%.2fM flops=%.2fM acc=%.4f",pre_params / 1e6, pre_flops / 1e6, pre_acc)
    report = {"before": {"params": pre_params, "flops": pre_flops, "acc": pre_acc}, "rounds": []}
    pcfg = cfg["pruning"]
    for round_idx in range(pcfg["num_prune_rounds"]):
        log.info("Pruning round %d/%d ...", round_idx + 1, pcfg["num_prune_rounds"])
        pruner = build_pruner(model, example_inputs, pcfg["prune_ratio_per_step"])
        pruner.step()  # physically removes channels network-wide
        params_after_prune = count_params(model)
        flops_after_prune = count_flops(model, input_shape)
        acc_after_prune = evaluate(model, test_loader, device)
        log.info("  post-prune (pre-finetune): params=%.2fM flops=%.2fM acc=%.4f",params_after_prune / 1e6, flops_after_prune / 1e6, acc_after_prune)
        acc_after_finetune = finetune(model, cfg, train_loader, test_loader, device,epochs=pcfg["finetune_epochs_per_round"], lr=pcfg["finetune_lr"],)
        report["rounds"].append({ "round": round_idx + 1, "params": params_after_prune,"flops": flops_after_prune,"acc_pre_finetune": acc_after_prune,"acc_post_finetune": acc_after_finetune,})

    final_params = count_params(model)
    final_flops = count_flops(model, input_shape)
    final_acc = evaluate(model, test_loader, device)
    report["after"] = {"params": final_params, "flops": final_flops, "acc": final_acc}
    log.info("Final: params=%.2fM (%.1f%% of original) flops=%.2fM (%.1f%% of original) acc=%.4f",final_params / 1e6, 100 * final_params / pre_params,final_flops / 1e6, 100 * final_flops / pre_flops, final_acc)
    save_checkpoint(model, pcfg["checkpoint"], extra={"acc": final_acc})
    import os
    os.makedirs("results", exist_ok=True)
    with open(pcfg["report_path"], "w") as f:
        json.dump(report, f, indent=2)
    return model, report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_pruning(cfg)