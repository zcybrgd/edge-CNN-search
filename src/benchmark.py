import argparse
import json
import os
import time
import yaml
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
from utils import setup_logging
log = setup_logging(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_fp16_engine(onnx_path: str, engine_path: str, workspace_gb: int = 1):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Build failed for {onnx_path}")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)


def benchmark(engine_path: str, input_shape: tuple, n_warmup: int = 30, n_iters: int = 300) -> dict:
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    d_input = cuda.mem_alloc(input_data.nbytes)
    output_shape = tuple(context.get_tensor_shape(output_name))
    d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, input_data, stream)
    for _ in range(n_warmup):
        context.execute_async_v3(stream.handle)
    stream.synchronize()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        context.execute_async_v3(stream.handle)
        stream.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"engine": engine_path,"mean_latency_ms": float(times.mean()),"p99_ms": float(np.percentile(times, 99)),"throughput_fps": float(1000 / times.mean()),}

def main(cfg: dict):
    input_size = cfg["data"]["input_size"]
    input_shape = (1, 3, input_size, input_size)
    configs = [("distilled_unpruned", "student_pruned.onnx"),  # export unpruned checkpoint separately fpr comparison
    ("final_pruned", "student_pruned_sim.onnx"),]
    results = []
    for label, onnx_path in configs:
        if not os.path.exists(onnx_path):
            log.warning("Skipping %s, %s not found", label, onnx_path)
            continue
        engine_path = f"engines/{label}.engine"
        build_fp16_engine(onnx_path, engine_path)
        r = benchmark(engine_path, input_shape)
        r["label"] = label
        results.append(r)
        log.info("%s: %.3f ms mean, %.1f FPS", label, r["mean_latency_ms"], r["throughput_fps"])
    os.makedirs("results", exist_ok=True)
    with open("results/jetson_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)