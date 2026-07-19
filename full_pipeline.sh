#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Training teacher..."
python3 src/teacher_model.py

echo "[2/6] Running evolutionary architecture search..."
python3 src/evolutionary_search.py

echo "[3/6] Distilling teacher into searched student..."
python3 src/distillation.py

echo "[4/6] Structured pruning + fine-tuning..."
python3 src/pruning.py

echo "[5/6] Exporting pruned model to ONNX..."
python3 src/export_onnx.py

echo "[6/6] Benchmarking on Jetson (run this step ON the Jetson device)..."
python3 src/benchmark_jetson.py

echo "Pipeline complete. Check results/ for all reports."