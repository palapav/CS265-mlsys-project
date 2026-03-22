# Midway Check-in Report  
CS265 Systems Project: Activation Checkpointing (w/o AC Baseline)

### Aditya Palaparthi
### GitHub Repo Link: https://github.com/palapav/CS265-mlsys-project

---

## 1. Introduction

This midway check-in completes **Phase 1 (Graph Profiler)** and reports the required **w/o AC** baseline experiments from `PROBLEM.md` deliverables **4(a)** and **4(b)**. The implementation in `graph_prof.py` profiles a full training iteration graph (forward + backward + optimizer), computes per-node runtime/memory statistics, performs static activation lifetime analysis (last forward use and first backward use), and produces peak-memory summaries. Experiments are automated in `midway_checkin.py` for `ResNet-152` and `BERT-Base`, with raw outputs in `midway_results_wo_ac.json`, the peak-memory-vs-batch graph in `midway_peak_memory_wo_ac.png`, and a dedicated category-breakdown graph at peak in `midway_peak_breakdown_at_peak_wo_ac.png`.

---

## 2. Problems Tackled

- `[Full-Iteration Graph Profiling]` Build a profiler that runs a traced graph node-by-node and captures compute/memory metrics over forward, backward, and optimizer phases.
- `[Forward/Backward Boundary Detection]` Reliably separate forward and backward regions in the traced graph to support static activation analysis.
- `[Activation Lifetime Analysis]` Identify activation tensors and compute each activation's last forward use and first backward use.
- `[Memory Breakdown at Peak]` Track a live-tensor memory model and break peak memory into semantic categories (`PARAM`, `ACT`, `GRAD`, `OPT_STATE`, `OTHER`).
- `[Midway Deliverables Automation]` Run repeatable baseline experiments across mini-batch sizes, export machine-readable results, and generate the required peak-memory bar graph (w/o AC).

---

## 3. Technical Description

### Problem/Solution 1: Full-iteration graph profiling

- **a) Problem framing:**  
  Midway deliverable 4(a) requires operator-level profiling statistics for training, not just forward inference. The challenge is that a standard model call does not expose a single explicit graph containing forward, backward, and optimizer updates in a form that can be profiled node-by-node.

- **b) High-level solution:**  
  Use the project compiler/tracer (`graph_tracer.compile`) to trace a full `train_step` and apply a custom transformation that instantiates `GraphProfiler` (subclassing `torch.fx.Interpreter`). The profiler executes nodes in topological order and records runtime and memory statistics for each node.

- **c) Deeper details:**  
  `GraphProfiler.run_node()` uses CUDA events for timing (`start_event`/`end_event`) and records memory deltas via `torch.cuda.memory_allocated()`. `GraphProfiler.run()` also tracks `torch.cuda.max_memory_allocated()` per profiled iteration and aggregates across profile runs. Warmup/profile scheduling is handled in `midway_checkin.py` via `warmup_iters=1`, `profile_iters=2`.

### Problem/Solution 2: Forward/backward separation and activation lifetime analysis

- **a) Problem framing:**  
  Activation checkpointing depends on knowing when each activation is last needed in forward and first needed in backward. Without explicit phase boundaries, static lifetime analysis is unreliable.

- **b) High-level solution:**  
  Insert separator ops in the train step (`SEPFunction.apply`) and locate them in the FX graph (`separator.sep.default` and `separator.sep_backward.default`) to define forward and backward boundaries.

- **c) Deeper details:**  
  In `GraphProfiler._analyze_activations()`, a node is treated as an activation candidate if it is produced in forward and has at least one backward user. The profiler stores:
  - `last_forward_use[node]`: latest forward user by node index.
  - `first_backward_use[node]`: earliest backward user by node index.  
  These statistics are exported in `largest_activations` for direct inspection.

### Problem/Solution 3: Peak memory modeling and category breakdown

- **a) Problem framing:**  
  Deliverable 4(a) asks for memory profiling and analysis, not just a single scalar peak. We need interpretable category-level breakdown at peak.

- **b) High-level solution:**  
  Maintain a live-tensor map while interpreting the graph. On each node, compute output tensor bytes, decrement input-use counters, remove dead tensors, and update current/peak totals.

- **c) Deeper details:**  
  The profiler now exports two complementary category views:
  - `peak_breakdown_live_mib`: category composition **at the instant of total live-memory peak** (sums exactly to `peak_live_mib`).
  - `peak_category_max_live_mib`: per-category maxima over the whole iteration (their sum can exceed total because peaks happen at different times).  
  For decomposed Adam graphs (foreach/copy form), placeholder roles are inferred from runtime tensor identity and categorized as `PARAM`, `OPT_STATE`, or `OTHER`; backward-compute-region nodes are categorized as `GRAD`; forward nodes with backward users are categorized as `ACT`.

### Problem/Solution 4: Midway experiments and required artifacts (w/o AC)

- **a) Problem framing:**  
  Midway requires (i) profiling + static analysis statistics and (ii) peak memory vs mini-batch size bar graph, both **without activation checkpointing**.

- **b) High-level solution:**  
  Run `ResNet-152` and `BERT-Base` baselines across specified batch sizes in `midway_checkin.py`, collect profiler summaries into JSON, and render a peak-memory bar graph from those summaries.

- **c) Deeper details:**  
  - `ResNet-152` batch sizes: `2, 4, 8`
  - `BERT-Base` batch sizes: `1, 2, 4, 8` (`seq_len=512`)  
  Results are serialized in `midway_results_wo_ac.json`; figure is saved to `midway_peak_memory_wo_ac.png`.

---

### Verified Midway Deliverables (w/o AC)

#### Check-in status

- Phase 1 completed (DONE).
- Deliverable 4(a): computation/memory profiling + static activation analysis. (DONE)
- Deliverable 4(b): peak memory vs mini-batch size graph (w/o AC). (DONE)
- Phase-1 peak memory breakdown graph from collected stats included. (DONE)

#### Experimental setup used for verification

- Device: `NVIDIA H100 80GB HBM3`
- Seed: `torch.manual_seed(0)`
- Profiler schedule: `1` warmup + `2` profile iterations
- Models: `ResNet-152`, `BERT-Base`
- Verification run command:

```bash
srun --partition=pi_faez --gres=gpu:h100:1 --cpus-per-task=8 --mem=128G --time=02:00:00 \
  bash -c "source /orcd/compute/faez/001/miniforge3/etc/profile.d/conda.sh && conda activate cs265 && cd /home/apalapar/projects/CS265-mlsys-project && python midway_checkin.py"
```

#### Deliverable 4(a): profiling statistics and static analysis

Baseline summary configurations (smallest successful batch for each model):

| Model | Batch | Graph nodes | Activations | Peak CUDA MiB | Peak live MiB |
|---|---:|---:|---:|---:|---:|
| ResNet-152 | 2 | 18498 | 777 | 1214.87 | 1377.70 |
| BERT-Base | 1 | 8810 | 353 | 2312.67 | 2954.15 |

Peak live-memory breakdown at total peak (MiB, sums exactly to `peak_live_mib`):

| Model | Batch | PARAM | ACT | GRAD | OPT_STATE | OTHER |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-152 | 2 | 229.62 | 0.00 | 229.62 | 459.23 | 459.23 |
| BERT-Base | 1 | 507.30 | 0.00 | 417.76 | 1014.60 | 1014.48 |

Note: for these baseline configurations, the model's total live-memory peak occurs in optimizer-related portions of the iteration; activation contribution at that exact timestamp is therefore `0.00` even though activation maxima are substantial (see table below).

Peak memory breakdown graph (stacked bars at total-peak timestamp, baseline configs):

![Peak live-memory breakdown at total peak (w/o AC)](midway_peak_breakdown_at_peak_wo_ac.png)

Per-category maxima over full iteration (MiB, **do not** sum to total peak):

| Model | Batch | PARAM | ACT | GRAD | OPT_STATE | OTHER |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-152 | 2 | 229.62 | 345.67 | 247.67 | 459.24 | 688.85 |
| BERT-Base | 1 | 507.30 | 834.55 | 507.18 | 1014.60 | 1521.78 |

Top runtime operators (avg ms):

- **ResNet-152 (batch 2):** `_foreach_div_1` (3.15), `_foreach_div_2` (3.04), `_foreach_addcdiv` (2.15), `_foreach_addcmul` (1.86), `_foreach_mul` (1.28)
- **BERT-Base (batch 1):** `_foreach_addcdiv` (1.53), `_foreach_div_2` (1.44), `_foreach_div_1` (1.41), `_foreach_addcmul` (1.32), `_foreach_sqrt_1` (0.92)

Largest activations with lifetime boundaries (size MiB, `last_fwd -> first_bwd`):

- **ResNet-152 (batch 2):**
  - `t` (7.81, `addmm -> t_1`)
  - `convolution_7` (6.12, `cudnn_batch_norm_7 -> cudnn_batch_norm_backward_147`)
  - `convolution_10` (6.12, `cudnn_batch_norm_10 -> cudnn_batch_norm_backward_144`)
- **BERT-Base (batch 1):**
  - `t_73` (89.42, `addmm_73 -> t_74`)
  - `_log_softmax` (59.61, `nll_loss_forward -> nll_loss_backward`)
  - `t_46` (9.00, `addmm_46 -> t_182`)

Graph-level evidence (baseline configs):

- **Placeholder role categorization:**  
  - ResNet-152 batch 2: `param=467`, `opt_state=1401`, `other=469`, `total=2337`  
  - BERT-Base batch 1: `param=204`, `opt_state=612`, `other=6`, `total=822`
- **Boundary indices from traced graph:**  
  - ResNet-152 batch 2: `forward_end_idx=3634`, `backward_start_idx=3637`, `optimizer_start_idx=4937`
  - BERT-Base batch 1: `forward_end_idx=1505`, `backward_start_idx=1508`, `optimizer_start_idx=2876`
- **Region node counts:**  
  - ResNet-152 batch 2: `forward=3635`, `backward_compute=1300`, `optimizer=13561`
  - BERT-Base batch 1: `forward=1506`, `backward_compute=1368`, `optimizer=5934`
- **Separator validation checks:** all `true` for both models (`forward_separator_found`, `backward_separator_found`, `forward_before_backward`, `optimizer_after_backward`).

Representative node-trace evidence around boundaries:

```text
ResNet-152 (batch 2):
idx 3634: separator.sep.default
idx 3637: separator.sep_backward.default
idx 4937: aten._foreach_add.Scalar   (optimizer starts)

BERT-Base (batch 1):
idx 1505: separator.sep.default
idx 1508: separator.sep_backward.default
idx 2876: aten._foreach_add.Scalar   (optimizer starts)
```

Raw experiment artifact (full JSON):
- [`midway_results_wo_ac.json`](midway_results_wo_ac.json)

Preview excerpt from `midway_results_wo_ac.json`:

```json
{
  "phase_1_completed": true,
  "deliverable_4a_without_ac": {
    "description": "Compute/memory profiling statistics and static activation analysis",
    "results": {
      "ResNet-152": [
        {
          "batch_size": 2,
          "profile": {
            "num_graph_nodes": 18498,
            "num_activations": 777,
            "peak_live_mib": 1377.7014083862305,
            "peak_cuda_mib": 1214.86572265625,
            "peak_breakdown_live_mib": {
              "param": 229.61734008789062,
              "act": 0.0,
              "grad": 229.61734008789062,
              "opt_state": 459.23468017578125,
              "other": 459.23204803466797
            },
            "graph_evidence": {
              "boundary_indices": {
                "forward_end_idx": 3634,
                "backward_start_idx": 3637,
                "optimizer_start_idx": 4937
              }
            }
          }
        }
      ],
      "BERT-Base": [
        {
          "batch_size": 1,
          "profile": {
            "num_graph_nodes": 8810,
            "num_activations": 353,
            "peak_live_mib": 2954.1454887390137,
            "peak_cuda_mib": 2312.67431640625
          }
        }
      ]
    }
  }
}
```

#### Deliverable 4(b): peak memory vs mini-batch size (w/o AC)

The bar graph is provided in `midway_peak_memory_wo_ac.png`. Values used:

| Model | Batch size | Peak CUDA MiB | Peak live MiB |
|---|---:|---:|---:|
| ResNet-152 | 2 | 1214.87 | 1377.70 |
| ResNet-152 | 4 | 1485.58 | 1423.13 |
| ResNet-152 | 8 | 2160.49 | 2091.52 |
| BERT-Base | 1 | 2312.67 | 2954.15 |
| BERT-Base | 2 | 2307.34 | 2954.15 |
| BERT-Base | 4 | 3239.38 | 3851.54 |
| BERT-Base | 8 | 5166.99 | 5765.51 |

Embedded figure (`midway_peak_memory_wo_ac.png`):

![Peak memory consumption vs mini-batch size (w/o AC)](midway_peak_memory_wo_ac.png)

Interpretation:
- Memory usage grows with batch size overall for both models.
- BERT has a much larger activation footprint than ResNet at comparable small-batch settings.
- The minor non-monotonicity between BERT batch `1` and `2` in CUDA peak is small and attributable to allocator/workspace behavior; trend is strongly increasing from batch `2` onward.
- `peak_breakdown_live_mib` is now a true decomposition at the total-peak timestamp; `peak_category_max_live_mib` is reported separately for per-category maxima over time.

---

## 4. Challenges

- Decomposed optimizer graphs are structurally different from fused-optimizer graphs; robust role assignment required runtime tensor identity matching.
- Peak-memory reporting needs two views (composition at total peak vs per-category maxima) to avoid misleading interpretations.
- Reducing profiling noise with short runs (`2` profile iterations) without making experimentation prohibitively slow.
- Preparing for Phase 2/3 integration: using static activation lifetime data to drive recomputation decisions and graph rewrites while maintaining gradient correctness.