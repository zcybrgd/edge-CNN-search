SlimNet — Full Problem Catalog + Toy-vs-Real Breakdown, Stage by Stage

You've hit a handful of these already. This is the complete list — problems you've seen, plus ones you likely haven't yet (search plateaus further, distillation not helping as much as hoped, pruning collapse, export/deployment issues) — organized by pipeline stage, each with: what it looks like, why it happens, and what industry actually does about it.

Stage 1: Teacher Training
Problems that can arise

P1.1 — Teacher overfits before student ever sees it.
Symptom: training loss keeps dropping, test accuracy plateaus or drops. You didn't hit this (your curve was clean), but it's common with small datasets or too many epochs without regularization.
Industry response: early stopping on validation accuracy (not just tracking the best checkpoint, actually stopping training), stronger augmentation (mixup, cutmix, randaugment), label smoothing. Some teams deliberately train the teacher slightly past its peak because a slightly-overfit but confident teacher can still distill well if temperature is tuned — this is a real, debated point in the KD literature, not settled.

P1.2 — Teacher is accurate but poorly calibrated (overconfident).
Symptom: teacher's softmax outputs are near one-hot even when wrong — very low entropy. This starves distillation of "dark knowledge" regardless of temperature.
Industry response: check teacher calibration explicitly (Expected Calibration Error, reliability diagrams) before ever starting distillation, not after a disappointing KD result. If poorly calibrated, either apply temperature scaling as a post-hoc calibration step on the teacher itself, or retrain with label smoothing, which tends to produce better-calibrated softmax outputs.

P1.3 — Architecture/input mismatch (the CIFAR stem fix you already applied).
Symptom: standard ImageNet architectures (7×7 stride-2 stem + maxpool) destroy small images before the network does any real work — accuracy caps out low no matter how long you train.
Industry response: exactly what you did — adapt the stem for the target input resolution. This is standard practice whenever reusing an ImageNet-designed architecture on a different input size; teams maintain small "adapter" variants of common backbones for exactly this reason.

Stage 2: Architecture Search
Problems that can arise

P2.1 — Proxy ranking doesn't correlate with final ranking.
This is the big one, and you've already seen a mild version of it (the gen-9 crash). At scale, the real risk isn't one bad candidate — it's that the overall ranking from 3-epoch proxies could be systematically wrong, i.e., the architecture that looks best at 3 epochs isn't the one that would've been best at 100.
Industry response: this is a well-studied problem called proxy-target correlation, usually measured with Kendall's Tau or Spearman correlation between proxy ranking and a sample of full-training ground truth. Real NAS teams periodically spot-check: take the top-5 candidates from a cheap search, fully train just those 5, and confirm the proxy's #1 pick is actually competitive. If correlation is poor, they either extend the proxy budget, switch to a different proxy (e.g., "zero-cost" proxies based on gradient statistics at initialization, which are even cheaper AND sometimes correlate better than a few epochs of training), or add weight-sharing (ENAS/DARTS-style supernets) which avoids training each candidate from scratch entirely.

P2.2 — Search collapses to one region of the space (premature convergence).
Symptom: after enough generations, every candidate looks nearly identical — the population lost diversity too early and stopped exploring.
Industry response: increase mutation rate, use larger tournament sizes relative to population (less aggressive selection pressure), or explicitly track population diversity (e.g. average pairwise architecture distance) and inject fresh random candidates if diversity drops below a threshold. Some systems use multiple parallel populations ("islands") that occasionally exchange candidates, specifically to avoid this failure mode.

P2.3 — Search space itself excludes the actual best architecture.
This is invisible from inside the search — if your fixed choices (block type, stride schedule, stage count) were wrong for this task, no amount of searching within the space finds a good answer, because it isn't there.
Industry response: sanity-check the search space boundaries before trusting search results — try the extremes (smallest allowed config, largest allowed config) by hand, and compare against a well-known hand-designed baseline (e.g., actual MobileNetV2-0.5x) at similar FLOPs. If the hand-designed baseline beats everything your search found, the space is likely too constrained or mis-specified, not that the search failed.

P2.4 — Search takes too long / compute budget exceeded mid-run.
Industry response: checkpointing the search state itself (population + log), not just final results, so a crashed or interrupted search can resume rather than restart. Your current search_log.json accumulates but the population state itself isn't checkpointed for resume — worth adding if you ever run a much longer search.

Stage 3: Distillation
Problems that can arise

P3.1 — Capacity gap: student too small to benefit from the teacher.
This is the leading hypothesis for your "modest" +2.67 point gain. Beyond a certain teacher/student size ratio, the student structurally can't represent the teacher's decision boundary well enough for KD to transfer much.
Industry response: this is documented in the literature as the "capacity gap problem" in knowledge distillation. Standard fixes: (a) teacher assistant distillation — distill from teacher → medium-sized intermediate model → small student, in stages, rather than one giant jump; (b) shrink the gap by choosing a smaller/weaker teacher deliberately if the deployment target is very small; (c) accept the gap and optimize hyperparameters (temperature/alpha) as far as they'll go, which is exactly the sweep you're running now.

P3.2 — Distillation loss diverges or the student ignores the teacher entirely.
Symptom: soft loss term stays flat or the student's outputs never move toward the teacher's distribution — often caused by a temperature/alpha combination where the hard-label term dominates so much the soft term is negligible.
Industry response: log the two loss components (soft vs hard) separately during training, not just the combined loss — if one term is orders of magnitude smaller than the other, it's being drowned out regardless of alpha's nominal weighting. This is a common instrumentation gap; teams that skip separate loss logging often waste time tuning alpha when the real issue is a scale mismatch between the two loss terms.

P3.3 — Feature-level mismatch (relevant if you ever go beyond logit distillation).
Not something your current pipeline does, but worth knowing: some KD variants also match intermediate feature maps (not just final logits) between teacher and student. This requires the two networks to have compatible intermediate shapes, which student/teacher of very different architectures (like yours — MobileNet-style vs ResNet50) often don't have naturally.
Industry response: insert learned 1×1 "adapter" convolutions purely to reshape student features to match teacher feature dimensions for the loss computation, discarded at inference time. This is a legitimate next-level extension if logit-only distillation plateaus.

P3.4 — Overfitting to the teacher's mistakes.
If the teacher is systematically wrong on some class or pattern, a student trained heavily on soft labels (high alpha) can inherit that specific error mode more strongly than a student trained on hard labels alone would.
Industry response: monitor student accuracy on a held-out set where teacher accuracy is known to be weak, if such a subset is identifiable; in general this is one more reason alpha is swept rather than pinned to a high value by default.

Stage 4: Pruning
Problems that can arise (several you've already lived through)

P4.1 — Post-prune accuracy collapse. (You saw this every round.)
Industry response: gradual pruning + fine-tune, exactly what you did — this is standard, not a workaround. Beyond that, teams sometimes use sensitivity analysis first: prune each layer individually by a small amount, measure the accuracy drop per layer, and use that to set a non-uniform pruning ratio (prune redundant layers more aggressively, sensitive layers less), rather than your current uniform 15%-per-round-everywhere approach. This is a real improvement worth trying if you want to push pruning further with less accuracy cost.

P4.2 — Diminishing fine-tuning returns / accuracy cost is fundamental. (You tested this directly — 10 vs 20 epochs.)
Industry response: when more fine-tuning doesn't help, the next lever is a gentler ratio per round (e.g. 8-10% instead of 15%) with the same total number of rounds, or a different importance metric (see below) rather than more training time on the current metric's choices.

P4.3 — L1-norm importance is a weak heuristic for some layer types.
Depthwise convolution layers (your entire student architecture is built from them) behave differently than standard convs under L1-norm pruning — a depthwise filter's magnitude doesn't map to "importance" as cleanly as in a standard conv, because each channel is independent (no cross-channel mixing to smooth out an unlucky low-magnitude filter). This is a known, published limitation specifically for MobileNet-style architectures under naive magnitude pruning.
Industry response: use gradient-based or Taylor-expansion importance scores (estimate each channel's contribution to the loss, not just its weight magnitude — requires one backward pass over a calibration batch, more expensive per pruning decision but often meaningfully better for exactly this architecture family), or skip pruning the depthwise layers preferentially and focus removal on the pointwise (1×1) layers where channel importance is more standard-conv-like.

P4.4 — Residual/skip-connection shape conflicts during structured pruning.
If two branches feeding a residual add get pruned independently by different amounts, their shapes stop matching and the add operation breaks.
Industry response: exactly why torch-pruning's dependency-graph tracing exists — it automatically constrains grouped/coupled layers (like both branches of a residual add) to be pruned by the same amount together. If you ever hand-roll pruning without a dependency-aware library, this is the single most common cause of silent correctness bugs (the model runs, but the wrong channels get wired together).

P4.5 — Pruned model's BatchNorm statistics are stale.
After removing channels, remaining BatchNorm running mean/variance were computed including the now-deleted channels' influence on the layer before it — technically fine after fine-tuning re-updates the running stats, but if fine-tuning is very short, BN stats might not have re-stabilized yet, causing eval-mode accuracy to look worse than train-mode accuracy would suggest.
Industry response: some pruning pipelines explicitly reset and "recalibrate" BatchNorm statistics with a few forward passes (no gradient updates) on training data immediately after pruning, before starting the real fine-tuning — a cheap step that can meaningfully improve the pre-fine-tune baseline your logs show as a near-total crash.

Stage 5: Export + Deployment (you haven't hit these yet — here's what's coming)

P5.1 — ONNX export silently drops or misrepresents an operation.
Some PyTorch ops don't have a clean ONNX equivalent, or the exporter picks an ONNX op that behaves subtly differently (common with certain padding modes, or custom activation functions). Your architecture uses only standard ops (Conv2d, BatchNorm2d, ReLU6, AdaptiveAvgPool2d, Linear), which are all well-supported — low risk here, but worth knowing.
Industry response: always run the parity check (verify_onnx.py-style comparison against the original PyTorch output) before trusting an export — never assume export succeeded just because it didn't error.

P5.2 — TensorRT doesn't support an ONNX op / falls back or fails to build.
Industry response: check the build logs closely — TensorRT logs warnings when it can't fuse an operation or falls back to a slower implementation path. If a build fails outright, the standard fix is either simplifying the offending op in the original PyTorch model (swap for a TensorRT-friendly equivalent) or, for teams with more resources, writing a custom TensorRT plugin for the unsupported op — a meaningfully bigger undertaking, usually avoided unless the op is unavoidable.

P5.3 — Engine builds fine but latency is worse than expected given the FLOP reduction.
This is the "FLOPs isn't the same as latency" lesson from EdgeBench resurfacing. A pruned model with fewer FLOPs can still be slower than expected if the remaining channel counts aren't hardware-tile-friendly (not multiples of 8, as your _make_divisible function is supposed to prevent) — but pruning can undo that alignment, since it removes an arbitrary number of channels based on importance ranking, not on staying divisible by 8.
Industry response: after structured pruning, explicitly round the resulting channel counts to the nearest hardware-friendly multiple (may mean keeping a few extra "unimportant" channels just to hit a clean tile boundary) — a real, common post-pruning step that's easy to forget. Worth checking your final pruned model's actual channel counts against this.

P5.4 — Jetson runs out of memory during engine build or benchmark.
Jetson devices have unified memory (shared between CPU and GPU) and much less of it than a desktop GPU — a workspace size that's fine on your GTX 1650 Ti can OOM on Jetson.
Industry response: reduce workspace_gb in the TensorRT builder config explicitly for embedded targets, monitor with tegrastats during build (not just during inference), and build engines with --fp16 rather than attempting INT8 calibration on-device if memory is tight, since calibration requires holding a calibration dataset and intermediate activations simultaneously.

P5.5 — Batch size 1 is much less efficient than the model's "ideal" utilization point.
Real-time edge inference is almost always batch size 1 (one frame/sample at a time), but GPUs (including Jetson's) are typically most efficient at somewhat larger batches — small-batch inference under-utilizes the hardware.
Industry response: this is accepted as an inherent cost of real-time edge deployment, not something to "fix" — the alternative (batching multiple inputs to increase throughput) directly trades away latency, which is usually the actual constraint edge deployments care about. Some systems batch across independent camera streams or sensor inputs when multiple exist on the same device, to recover some efficiency without hurting per-stream latency.

"This project implements the real algorithmic skeleton of architecture search, knowledge distillation, and structured pruning — the same techniques industry uses — but at a compute and rigor budget appropriate for one person on one GPU: smaller search spaces, cheaper proxies, fewer statistical repeats, and a single hardware target, rather than the massively parallel, statistically-averaged, multi-device validation a production ML platform team would run."

# The Complete "What's Toy Here" Table

| Component | What's toy about it | What real industry version looks like |
|---|---|---|
| **Architecture search** | ~10^9-point space constrained to one block family; 3-epoch proxy fitness; 35 total candidates evaluated | Larger/less-constrained spaces (arbitrary connectivity), thousands of candidates, near-convergence training or weight-sharing supernets, sometimes hardware-latency-aware search (measuring real device latency during search, not FLOP proxies) |
| **Teacher-student pair** | Single teacher, single student size, single dataset | Multiple teacher/student size ratios tested; teacher assistant chains for large capacity gaps; multi-dataset validation before trusting a KD recipe |
| **Hyperparameter tuning** | Manual default + a small grid sweep (9 combos) | Automated hyperparameter optimization (Bayesian optimization, population-based training) across a much larger space, often run continuously as part of an ML platform rather than a one-off script |
| **Pruning importance metric** | L1-norm weight magnitude only | Often gradient/Taylor-based, sometimes learned importance (a small auxiliary network predicts channel importance), combined with sensitivity analysis for non-uniform per-layer ratios |
| **Pruning schedule** | Fixed 15%-per-round, 3 rounds, uniform across all layers | Per-layer sensitivity-informed ratios; sometimes continuous/differentiable pruning (e.g. learnable channel gates trained end-to-end) rather than discrete prune-then-finetune rounds |
| **Hardware target validation** | One device (Jetson Orin Nano), one benchmark run | Testing across a device matrix (multiple Jetson SKUs, multiple TensorRT versions), statistical latency measurement (many runs, reporting distributions not single numbers), sometimes automated regression testing so a code change that regresses latency is caught before deployment |
| **Statistical rigor** | Single run per configuration, single seed | Multiple seeds averaged with reported variance, for every stage — search, distillation, pruning — because any single number here could partly be seed luck rather than a real effect |