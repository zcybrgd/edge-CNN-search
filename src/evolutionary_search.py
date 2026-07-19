import argparse
import random
import yaml
import torch
import torch.nn as nn
from search_space import sample_random_arch, mutate_arch, ArchConfig
from student_model import build_student
from data import get_dataloaders
from utils import setup_logging, set_seed, count_params, count_flops, append_json_log

log = setup_logging(__name__)


def proxy_train_and_eval(arch: ArchConfig, cfg: dict, train_loader, test_loader, device) -> float:
    """Trains a candidate for a small number of epochs and returns test accuracy as the fitness signal"""
    model = build_student(arch, cfg["data"]["num_classes"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg["search"]["proxy_train_epochs"]):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


class Candidate:
    def __init__(self, arch: ArchConfig, fitness: float, params: int, flops: int):
        self.arch = arch
        self.fitness = fitness
        self.params = params
        self.flops = flops


def run_search(cfg: dict):
    scfg = cfg["search"]
    set_seed(scfg["seed"])
    rng = random.Random(scfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_dataloaders(cfg)
    log.info("Initializing population of %d random architectures:", scfg["population_size"])
    population = []
    for i in range(scfg["population_size"]):
        arch = sample_random_arch(rng)
        model = build_student(arch, cfg["data"]["num_classes"])
        params = count_params(model)
        flops = count_flops(model, (1, 3, cfg["data"]["input_size"], cfg["data"]["input_size"]))
        model.to(device)
        fitness = proxy_train_and_eval(arch, cfg, train_loader, test_loader, device)
        candidate = Candidate(arch, fitness, params, flops)
        population.append(candidate)
        entry = {"generation": 0, "candidate_index": i, "fitness": fitness,"params": params, "flops": flops, "arch": arch.to_dict(),}
        append_json_log(scfg["log_path"], entry)
        log.info("Init candidate %d: acc=%.4f params=%.2fM flops=%.2fM",i, fitness, params / 1e6, flops / 1e6)

    best_overall = max(population, key=lambda c: c.fitness)

    for gen in range(1, scfg["generations"] + 1):
        tournament = rng.sample(population, scfg["tournament_size"])
        parent = max(tournament, key=lambda c: c.fitness)
        child_arch = mutate_arch(parent.arch, rng)
        model = build_student(child_arch, cfg["data"]["num_classes"])
        params = count_params(model)
        flops = count_flops(model, (1, 3, cfg["data"]["input_size"], cfg["data"]["input_size"]))
        model.to(device)
        fitness = proxy_train_and_eval(child_arch, cfg, train_loader, test_loader, device)
        child = Candidate(child_arch, fitness, params, flops)
        population.pop(0)
        population.append(child)
        entry = {"generation": gen, "fitness": fitness,"params": params, "flops": flops, "arch": child_arch.to_dict(),}
        append_json_log(scfg["log_path"], entry)
        log.info("Gen %d: child acc=%.4f params=%.2fM flops=%.2fM",gen, fitness, params / 1e6, flops / 1e6)
        if fitness > best_overall.fitness:
            best_overall = child
            log.info("New best architecture found at generation %d (acc=%.4f)", gen, fitness)
    log.info("Search complete. Best architecture: acc=%.4f, params=%.2fM, flops=%.2fM",best_overall.fitness, best_overall.params / 1e6, best_overall.flops / 1e6)
    import json, os
    os.makedirs(os.path.dirname(scfg["best_arch_path"]), exist_ok=True)
    with open(scfg["best_arch_path"], "w") as f:
        json.dump(best_overall.arch.to_dict(), f, indent=2)
    return best_overall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_search(cfg)

"""
Regularized evolutionary search over the SlimNet search space.
Algorithm: maintain a population; each generation, sample a tournament
subset, mutate the best member of the tournament, evaluate the child
with a cheap proxy training run, and replace the oldest population member 
"""