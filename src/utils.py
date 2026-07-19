import json
import logging
import os
import random
import numpy as np
import torch


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",)
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: torch.nn.Module, input_shape: tuple) -> int:
    from fvcore.nn import FlopCountAnalysis
    model.eval()
    dummy = torch.randn(*input_shape)
    device = next(model.parameters()).device
    dummy = dummy.to(device)
    with torch.no_grad():
        analysis = FlopCountAnalysis(model, dummy)
        analysis.unsupported_ops_warnings(False)
        total = analysis.total()
        print(f"FLOPs: {total / 1e9:.2f} GFLOPs")
    return int(total)


def save_checkpoint(model: torch.nn.Module, path: str, extra: dict = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str, device: str = "cpu") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    return payload


def append_json_log(path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log = []
    if os.path.exists(path):
        with open(path) as f:
            log = json.load(f)
    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)