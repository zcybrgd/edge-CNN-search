#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Training teacher"
python src/teacher_model.py

echo "[2/6] Running evolutionary architecture search"
python src/evolutionary_search.py

echo "[3/6] Distilling teacher into searched student"
python src/distillation.py

echo "[4/6] Structured pruning + fine-tuning"
python src/pruning.py

echo "[5/6] Exporting pruned model to ONNX"
python src/export_onnx.py

echo "[6/6] Benchmarking on Jetson"
python src/benchmark_jetson.py

echo "Pipeline complete. Check results/ for all reports"