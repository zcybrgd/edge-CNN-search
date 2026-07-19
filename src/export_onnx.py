import argparse
import json
import yaml
import torch
import onnx
from onnxsim import simplify
from student_model import build_student
from search_space import ArchConfig
from utils import setup_logging, load_checkpoint
log = setup_logging(__name__)


def export(cfg: dict, out_path: str = "student_pruned.onnx"):
    device = "cpu" 
    with open(cfg["search"]["best_arch_path"]) as f:
        arch = ArchConfig.from_dict(json.load(f))
    model = build_student(arch, cfg["data"]["num_classes"])
    load_checkpoint(model, cfg["pruning"]["checkpoint"], device=device)
    model.eval()
    input_size = cfg["data"]["input_size"]
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(model, dummy, out_path,input_names=["input"], output_names=["output"],opset_version=17,dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},)
    log.info("Exported %s", out_path)
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)  
    simplified, check = simplify(onnx_model)
    assert check, "ONNX simplification check failed"
    sim_path = out_path.replace(".onnx", "_sim.onnx")
    onnx.save(simplified, sim_path)
    log.info("Simplified and saved %s", sim_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    export(cfg)