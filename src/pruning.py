import argparse
import copy
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


def build_importance(importance_name: str):
    if importance_name == "taylor":
        return tp.importance.TaylorImportance()
    if importance_name == "magnitude":
        return tp.importance.MagnitudeImportance(p=1)
    raise ValueError(f"Unknown importance metric: {importance_name!r} (expected 'taylor' or 'magnitude')")


def accumulate_taylor_gradients(model: nn.Module, loader, device, num_batches: int):
    model.zero_grad()
    model.train()
    criterion = nn.CrossEntropyLoss()
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()
    model.eval()


def is_depthwise(m: nn.Module) -> bool:
    return isinstance(m, nn.Conv2d) and m.groups == m.in_channels == m.out_channels


def get_ignored_layers(model: nn.Module):
    ignored = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            ignored.append(m)  # never prune the final classifier layer
    return ignored


def get_prunable_convs(model: nn.Module, ignored_layers):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d) and m not in ignored_layers]

@torch.no_grad()
def _quick_eval(model, loader, device, num_batches):
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def run_sensitivity_analysis(model, cfg, example_inputs, calib_loader, device, probe_ratio: float, eval_batches: int, importance_name: str):
    ignored_layers = get_ignored_layers(model)
    prunable = get_prunable_convs(model, ignored_layers)
    base_acc = _quick_eval(model, calib_loader, device, eval_batches)
    log.info("Sensitivity analysis: probing %d conv layers at %.0f%% each (calib acc=%.4f)",len(prunable), probe_ratio * 100, base_acc)
    sensitivity = {}  # layer -> accuracy drop when pruned alone by probe_ratio
    for idx, layer in enumerate(prunable):
        probe_model = copy.deepcopy(model)
        # map the layer on the original model to the corresponding layer on the copy
        probe_layer = dict(zip(model.modules(), probe_model.modules()))[layer]
        probe_example = example_inputs.clone()
        importance = build_importance(importance_name)
        if importance_name == "taylor":
            accumulate_taylor_gradients(probe_model, calib_loader, device, num_batches=1)
        probe_ignored = [dict(zip(model.modules(), probe_model.modules()))[m] for m in ignored_layers]
        pruner = tp.pruner.MagnitudePruner(probe_model, probe_example, importance=importance,pruning_ratio=0.0, pruning_ratio_dict={probe_layer: probe_ratio},ignored_layers=probe_ignored,)
        try:
            pruner.step()
        except Exception as e:
            log.warning("  [%d/%d] layer could not be probed in isolation (%s); marking as sensitive",idx + 1, len(prunable), e)
            sensitivity[layer] = float("inf")
            continue
        probed_acc = _quick_eval(probe_model, calib_loader, device, eval_batches)
        drop = base_acc - probed_acc
        sensitivity[layer] = drop
        log.info("  [%d/%d] %s: acc drop=%.4f", idx + 1, len(prunable), layer.__class__.__name__, drop)
        del probe_model, pruner
    return sensitivity


def build_ratio_dict(model, sensitivity: dict, base_ratio: float, depthwise_ratio_scale: float, min_scale: float, max_scale: float):
    finite_drops = [d for d in sensitivity.values() if d != float("inf")]
    max_drop = max(finite_drops) if finite_drops else 1.0
    max_drop = max(max_drop, 1e-6)

    ratio_dict = {}
    for layer, drop in sensitivity.items():
        if drop == float("inf"):
            scale = min_scale
        else:
            # normalize drop to [0, 1], invert so redundant layers -> scale near max_scale
            norm_drop = min(drop / max_drop, 1.0)
            scale = max_scale - norm_drop * (max_scale - min_scale)
        if is_depthwise(layer):
            scale *= depthwise_ratio_scale
        ratio_dict[layer] = max(0.0, min(base_ratio * scale, 0.9))
    return ratio_dict


def build_pruner(model: nn.Module, example_inputs: torch.Tensor, pcfg: dict,
                  train_loader=None, device=None, sensitivity: dict = None):
    importance_name = pcfg.get("importance", "taylor")
    importance = build_importance(importance_name)
    ignored_layers = get_ignored_layers(model)

    if importance_name == "taylor":
        accumulate_taylor_gradients(model, train_loader, device,num_batches=pcfg.get("taylor_calib_batches", 4))

    base_ratio = pcfg["prune_ratio_per_step"]
    pruning_ratio_dict = None
    if sensitivity:
        pruning_ratio_dict = build_ratio_dict(model, sensitivity, base_ratio,depthwise_ratio_scale=pcfg.get("depthwise_ratio_scale", 0.4),min_scale=pcfg.get("sensitivity_min_scale", 0.3),max_scale=pcfg.get("sensitivity_max_scale", 1.3),)
        pruned_summary = {("depthwise" if is_depthwise(l) else "pointwise/other"): [] for l in pruning_ratio_dict}
        for l, r in pruning_ratio_dict.items():
            pruned_summary["depthwise" if is_depthwise(l) else "pointwise/other"].append(round(r, 3))
        for kind, ratios in pruned_summary.items():
            if ratios:
                log.info("  ratio dict (%s): mean=%.3f min=%.3f max=%.3f n=%d",kind, sum(ratios) / len(ratios), min(ratios), max(ratios), len(ratios))
    elif pcfg.get("depthwise_ratio_scale", 1.0) != 1.0:
        # even without full sensitivity analysis, dampen depthwise pruning (P4.3)
        dw_scale = pcfg.get("depthwise_ratio_scale", 0.4)
        pruning_ratio_dict = {
            m: base_ratio * dw_scale for m in get_prunable_convs(model, ignored_layers) if is_depthwise(m)
        }

    pruner = tp.pruner.MagnitudePruner(model, example_inputs, importance=importance,pruning_ratio=base_ratio, pruning_ratio_dict=pruning_ratio_dict,ignored_layers=ignored_layers,)
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

    use_sensitivity = pcfg.get("sensitivity_analysis", True)
    sensitivity = None
    if use_sensitivity:
        sensitivity = run_sensitivity_analysis(model, cfg, example_inputs, train_loader, device,probe_ratio=pcfg.get("sensitivity_probe_ratio", 0.05),eval_batches=pcfg.get("sensitivity_eval_batches", 5),importance_name=pcfg.get("importance", "taylor"),)
        report["sensitivity"] = {
            layer.__class__.__name__ + f"_{i}": (None if d == float("inf") else d)
            for i, (layer, d) in enumerate(sensitivity.items())
        }

    for round_idx in range(pcfg["num_prune_rounds"]):
        log.info("Pruning round %d/%d ...", round_idx + 1, pcfg["num_prune_rounds"])
        pruner = build_pruner(model, example_inputs, pcfg,train_loader=train_loader, device=device, sensitivity=sensitivity)
        pruner.step()  #physically removes channels network-wide
        params_after_prune = count_params(model)
        flops_after_prune = count_flops(model, input_shape)
        acc_after_prune = evaluate(model, test_loader, device)
        log.info("  post-prune (pre-finetune): params=%.2fM flops=%.2fM acc=%.4f",params_after_prune / 1e6, flops_after_prune / 1e6, acc_after_prune)
        acc_after_finetune = finetune(model, cfg, train_loader, test_loader, device,epochs=pcfg["finetune_epochs_per_round"], lr=pcfg["finetune_lr"],)
        report["rounds"].append({ "round": round_idx + 1, "params": params_after_prune,"flops": flops_after_prune,"acc_pre_finetune": acc_after_prune,"acc_post_finetune": acc_after_finetune,})
        if use_sensitivity and round_idx + 1 < pcfg["num_prune_rounds"]:
            sensitivity = run_sensitivity_analysis(model, cfg, example_inputs, train_loader, device,probe_ratio=pcfg.get("sensitivity_probe_ratio", 0.05),eval_batches=pcfg.get("sensitivity_eval_batches", 5),importance_name=pcfg.get("importance", "taylor"),)

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
