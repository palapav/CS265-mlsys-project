"""End-to-end final-deliverable experiment driver.

For each (model in {ResNet-152, BERT-Base}, batch_size) we:
  1. Trace the joint forward+backward+optimizer graph via `compile()`.
  2. Profile the baseline graph with `GraphProfiler` (Phase 1).
  3. Run mu-TWO-style `select_recomputations` on the baseline profile to pick
     activations to drop (Phase 2).
  4. Apply `apply_activation_checkpointing` to rewrite the graph (Phase 3).
  5. Re-profile the rewritten graph.
  6. Time both graphs end-to-end via CUDA events for iteration-latency
     measurement (deliverable 4(c)).

All raw data is dumped to `results/final/final_results.json` and the plots
required by deliverables 4(b)/4(c) are rendered into `results/final/`.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fx as fx

from activation_checkpoint import (
    apply_activation_checkpointing,
    select_recomputations,
)
from graph_prof import GraphProfiler, NodeType
from graph_tracer import compile
from midway_checkin import (
    _build_bert_inputs,
    _build_resnet152_inputs,
    _infer_placeholder_node_types,
    _init_optimizer_states,
    _to_mib,
    _validate_profile_summary,
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "final"
)
DEFAULT_OVERHEAD_RATIO = 0.5


@dataclass
class FinalExperimentConfig:
    model_name: str
    batch_sizes: List[int]


def _build_profile_summary(profiler: GraphProfiler) -> Dict[str, Any]:
    """Mirror the deliverable-4(a) summary shape from `midway_checkin.py`,
    plus a totals block that totals iteration runtime by region.
    """
    node_type_counts = Counter(profiler.node_types.values())

    forward_ms = sum(
        profiler.avg_runtime_ms.get(n, 0.0)
        for n in profiler.nodes
        if profiler._is_forward_region(n)
    )
    backward_ms = sum(
        profiler.avg_runtime_ms.get(n, 0.0)
        for n in profiler.nodes
        if profiler._is_backward_region(n)
    )
    optimizer_ms = sum(
        profiler.avg_runtime_ms.get(n, 0.0)
        for n in profiler.nodes
        if profiler._is_optimizer_region(n)
    )
    total_node_ms = sum(profiler.avg_runtime_ms.values())

    return {
        "num_graph_nodes": len(profiler.nodes),
        "num_activations": len(profiler.activation_nodes),
        "peak_live_mib": _to_mib(profiler.avg_peak_total_bytes),
        "peak_cuda_mib": _to_mib(profiler.avg_torch_peak_bytes),
        "node_type_counts": {
            "param": node_type_counts.get(NodeType.PARAM, 0),
            "act": node_type_counts.get(NodeType.ACT, 0),
            "grad": node_type_counts.get(NodeType.GRAD, 0),
            "opt_state": node_type_counts.get(NodeType.OPT_STATE, 0),
            "other": node_type_counts.get(NodeType.OTHER, 0),
        },
        "peak_breakdown_live_mib": {
            "param": _to_mib(profiler.avg_peak_by_type_bytes.get(NodeType.PARAM, 0.0)),
            "act": _to_mib(profiler.avg_peak_by_type_bytes.get(NodeType.ACT, 0.0)),
            "grad": _to_mib(profiler.avg_peak_by_type_bytes.get(NodeType.GRAD, 0.0)),
            "opt_state": _to_mib(
                profiler.avg_peak_by_type_bytes.get(NodeType.OPT_STATE, 0.0)
            ),
            "other": _to_mib(profiler.avg_peak_by_type_bytes.get(NodeType.OTHER, 0.0)),
        },
        "node_runtime_breakdown_ms": {
            "forward": float(forward_ms),
            "backward": float(backward_ms),
            "optimizer": float(optimizer_ms),
            "total_node_sum": float(total_node_ms),
        },
    }


def _measure_iter_latency_ms(
    gm: fx.GraphModule,
    args: Tuple[Any, ...],
    n_warmup: int = 3,
    n_iter: int = 10,
) -> Dict[str, float]:
    """End-to-end CUDA-event-based iteration latency. The graph already
    embeds optim.step() and zero_grad() so each call is one full training
    iteration.
    """
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(n_warmup):
            gm(*args)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    with torch.no_grad():
        for i in range(n_iter):
            starts[i].record()
            gm(*args)
            ends[i].record()
    torch.cuda.synchronize()

    samples = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {
        "n_iter": n_iter,
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "p10_ms": float(np.percentile(samples, 10)),
        "p90_ms": float(np.percentile(samples, 90)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "samples_ms": [float(x) for x in samples],
    }


def _profile_graph(
    gm: fx.GraphModule,
    args: Tuple[Any, ...],
    placeholder_types: Dict[fx.Node, NodeType],
    warmup_iters: int,
    profile_iters: int,
) -> GraphProfiler:
    profiler = GraphProfiler(gm, placeholder_node_types=placeholder_types)
    with torch.no_grad():
        for _ in range(warmup_iters):
            profiler.run(*args)
        profiler.reset_stats()
        for _ in range(profile_iters):
            profiler.run(*args)
    profiler.aggregate_stats()
    return profiler


def _run_one_setting(
    train_step_fn: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    example_inputs: Any,
    *,
    apply_ac: bool,
    overhead_ratio: float,
    warmup_iters: int = 1,
    profile_iters: int = 2,
) -> Dict[str, Any]:
    """Compile + (optionally) AC-rewrite + profile + time, all inside the
    `compile()` `graph_transformation` callback so we use the same tracing
    path as the starter pipeline.
    """
    bag: Dict[str, Any] = {"applied_ac": apply_ac}

    def graph_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        placeholder_types = _infer_placeholder_node_types(
            gm, args, model, optimizer
        )

        # 1. Profile baseline graph.
        baseline_profiler = _profile_graph(
            gm, args, placeholder_types, warmup_iters, profile_iters
        )
        bag["baseline_profile"] = _build_profile_summary(baseline_profiler)

        # 2. End-to-end iteration latency on baseline graph (CUDA events).
        bag["baseline_iter_latency"] = _measure_iter_latency_ms(gm, args)

        if not apply_ac:
            return gm

        # 3. Phase 2 selection on baseline profile.
        info = select_recomputations(
            baseline_profiler,
            max_recompute_overhead_ratio=overhead_ratio,
        )
        bag["selection"] = {
            "n_selected": len(info["selected"]),
            "selected_names": [n.name for n in info["selected"]],
            "total_bytes_saved_mib": float(info["total_bytes_saved_mib"]),
            "estimated_recompute_ms": float(info["total_recompute_ms"]),
            "overhead_budget_ms": float(info["overhead_budget_ms"]),
            "forward_runtime_ms": float(info["forward_runtime_ms"]),
            "overhead_ratio_used": overhead_ratio,
            "n_safe_candidates": int(info.get("n_safe_candidates", -1)),
            "n_skipped_unsafe_chain": int(info.get("n_skipped_unsafe_chain", -1)),
            "n_skipped_too_small": int(info.get("n_skipped_too_small", -1)),
        }
        if not info["selected"]:
            # Nothing to do; AC is a no-op.
            bag["ac_profile"] = bag["baseline_profile"]
            bag["ac_iter_latency"] = bag["baseline_iter_latency"]
            return gm

        # 4. Phase 3 rewrite.
        new_gm = apply_activation_checkpointing(gm, baseline_profiler, info["selected"])
        bag["rewritten_graph_node_count"] = len(list(new_gm.graph.nodes))

        # 5. Re-profile the rewritten graph.
        ac_profiler = _profile_graph(
            new_gm, args, placeholder_types, warmup_iters, profile_iters
        )
        bag["ac_profile"] = _build_profile_summary(ac_profiler)

        # 6. End-to-end iteration latency on rewritten graph.
        bag["ac_iter_latency"] = _measure_iter_latency_ms(new_gm, args)

        return new_gm

    compiled_fn = compile(train_step_fn, graph_transformation)
    compiled_fn(model, optimizer, example_inputs)
    return bag


def _safe_one_setting(
    *,
    model_name: str,
    batch_size: int,
    device: torch.device,
    apply_ac: bool,
    overhead_ratio: float,
) -> Dict[str, Any]:
    """Build inputs fresh and run one (model, batch_size, ac?) trial."""
    if model_name == "ResNet-152":
        model, optimizer, example_inputs, train_step = _build_resnet152_inputs(
            batch_size=batch_size, device=device
        )
    elif model_name == "BERT-Base":
        model, optimizer, example_inputs, train_step = _build_bert_inputs(
            batch_size=batch_size, device=device
        )
    else:
        raise ValueError(f"unknown model {model_name!r}")

    _init_optimizer_states(model, optimizer)
    bag = _run_one_setting(
        train_step,
        model,
        optimizer,
        example_inputs,
        apply_ac=apply_ac,
        overhead_ratio=overhead_ratio,
    )
    bag["model_name"] = model_name
    bag["batch_size"] = batch_size

    # Validate profiles' internal consistency exactly like the midway runner.
    if "baseline_profile" in bag:
        _validate_profile_summary(model_name, batch_size, bag["baseline_profile"])
    if "ac_profile" in bag:
        _validate_profile_summary(
            model_name + " (w/ AC)", batch_size, bag["ac_profile"]
        )

    # Tear down between trials so the CUDA allocator returns to a clean state.
    del model, optimizer, example_inputs, train_step
    torch.cuda.empty_cache()
    return bag


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_peak_memory_with_without_ac(
    results: Dict[str, List[Dict[str, Any]]], out_path: str
) -> None:
    """Deliverable 4(b): Peak-memory vs mini-batch-size, with vs. without AC."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=160)
    for ax_idx, model_name in enumerate(["ResNet-152", "BERT-Base"]):
        rows = results.get(model_name, [])
        if not rows:
            continue
        ax = axes[ax_idx]
        batch_sizes = [r["batch_size"] for r in rows]
        peak_no_ac = [r["baseline_profile"]["peak_cuda_mib"] for r in rows]
        peak_ac = [
            r.get("ac_profile", r["baseline_profile"])["peak_cuda_mib"]
            for r in rows
        ]
        x = np.arange(len(batch_sizes))
        bar_w = 0.4
        ax.bar(x - bar_w / 2, peak_no_ac, width=bar_w, label="w/o AC", color="#4C78A8")
        ax.bar(x + bar_w / 2, peak_ac, width=bar_w, label="w/ AC", color="#F58518")
        for i, (a, b) in enumerate(zip(peak_no_ac, peak_ac)):
            if a > 0:
                ax.text(
                    i + bar_w / 2,
                    b + max(peak_no_ac) * 0.01,
                    f"-{(a - b) / a * 100:.0f}%",
                    ha="center",
                    fontsize=8,
                    color="#444",
                )
        ax.set_xticks(x)
        ax.set_xticklabels([str(bs) for bs in batch_sizes])
        ax.set_xlabel("Mini-batch size")
        ax.set_ylabel("Peak CUDA memory (MiB)")
        ax.set_title(model_name)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
    plt.suptitle(
        "Deliverable 4(b): Peak GPU memory vs. mini-batch size, with and without AC",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_peak_breakdown_with_without_ac(
    results: Dict[str, List[Dict[str, Any]]], out_path: str
) -> None:
    """Stacked-bar comparison of memory composition at peak, with vs without AC.
    For each (model, batch_size), two bars are shown side by side."""
    categories = [
        ("param", "PARAM"),
        ("act", "ACT"),
        ("grad", "GRAD"),
        ("opt_state", "OPT_STATE"),
        ("other", "OTHER"),
    ]
    colors = {
        "param": "#4C78A8",
        "act": "#F58518",
        "grad": "#E45756",
        "opt_state": "#72B7B2",
        "other": "#54A24B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)
    for ax_idx, model_name in enumerate(["ResNet-152", "BERT-Base"]):
        rows = results.get(model_name, [])
        if not rows:
            continue
        ax = axes[ax_idx]
        n = len(rows)
        bar_w = 0.4
        x = np.arange(n)
        no_ac_x = x - bar_w / 2
        ac_x = x + bar_w / 2

        bottoms_no_ac = np.zeros(n)
        bottoms_ac = np.zeros(n)
        for key, pretty in categories:
            vals_no_ac = np.array(
                [r["baseline_profile"]["peak_breakdown_live_mib"].get(key, 0.0) for r in rows]
            )
            vals_ac = np.array(
                [
                    r.get("ac_profile", r["baseline_profile"])[
                        "peak_breakdown_live_mib"
                    ].get(key, 0.0)
                    for r in rows
                ]
            )
            ax.bar(
                no_ac_x,
                vals_no_ac,
                width=bar_w,
                bottom=bottoms_no_ac,
                color=colors[key],
                label=pretty if ax_idx == 0 else None,
            )
            ax.bar(
                ac_x,
                vals_ac,
                width=bar_w,
                bottom=bottoms_ac,
                color=colors[key],
                hatch="//",
            )
            bottoms_no_ac = bottoms_no_ac + vals_no_ac
            bottoms_ac = bottoms_ac + vals_ac

        ax.set_xticks(x)
        ax.set_xticklabels([str(r["batch_size"]) for r in rows])
        ax.set_xlabel("Mini-batch size  (left=w/o AC, right=w/ AC, hatched)")
        ax.set_ylabel("Live memory at peak (MiB)")
        ax.set_title(model_name)
        ax.grid(alpha=0.3, axis="y")
        if ax_idx == 0:
            ax.legend(ncols=3, fontsize=8, loc="upper left")
    plt.suptitle(
        "Peak live-memory breakdown by category, with and without AC", y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iter_latency_with_without_ac(
    results: Dict[str, List[Dict[str, Any]]], out_path: str
) -> None:
    """Deliverable 4(c): Iteration latency vs mini-batch-size, with vs without AC."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=160)
    for ax_idx, model_name in enumerate(["ResNet-152", "BERT-Base"]):
        rows = results.get(model_name, [])
        if not rows:
            continue
        ax = axes[ax_idx]
        batch_sizes = [r["batch_size"] for r in rows]
        no_ac_mean = [r["baseline_iter_latency"]["mean_ms"] for r in rows]
        ac_mean = [
            r.get("ac_iter_latency", r["baseline_iter_latency"])["mean_ms"] for r in rows
        ]
        no_ac_p10 = [r["baseline_iter_latency"]["p10_ms"] for r in rows]
        no_ac_p90 = [r["baseline_iter_latency"]["p90_ms"] for r in rows]
        ac_p10 = [
            r.get("ac_iter_latency", r["baseline_iter_latency"])["p10_ms"] for r in rows
        ]
        ac_p90 = [
            r.get("ac_iter_latency", r["baseline_iter_latency"])["p90_ms"] for r in rows
        ]
        x = np.arange(len(batch_sizes))
        bar_w = 0.4

        no_ac_err = [
            [m - lo for m, lo in zip(no_ac_mean, no_ac_p10)],
            [hi - m for m, hi in zip(no_ac_mean, no_ac_p90)],
        ]
        ac_err = [
            [m - lo for m, lo in zip(ac_mean, ac_p10)],
            [hi - m for m, hi in zip(ac_mean, ac_p90)],
        ]
        ax.bar(
            x - bar_w / 2,
            no_ac_mean,
            width=bar_w,
            yerr=no_ac_err,
            label="w/o AC",
            color="#4C78A8",
            capsize=3,
        )
        ax.bar(
            x + bar_w / 2,
            ac_mean,
            width=bar_w,
            yerr=ac_err,
            label="w/ AC",
            color="#F58518",
            capsize=3,
        )
        for i, (a, b) in enumerate(zip(no_ac_mean, ac_mean)):
            if a > 0:
                ax.text(
                    i + bar_w / 2,
                    b + max(no_ac_mean) * 0.02,
                    f"+{(b - a) / a * 100:.0f}%",
                    ha="center",
                    fontsize=8,
                    color="#444",
                )
        ax.set_xticks(x)
        ax.set_xticklabels([str(bs) for bs in batch_sizes])
        ax.set_xlabel("Mini-batch size")
        ax.set_ylabel("Iteration latency (ms)")
        ax.set_title(model_name)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
    plt.suptitle(
        "Deliverable 4(c): Iteration latency vs. mini-batch size, with and without AC",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_final_experiment(
    overhead_ratio: float = DEFAULT_OVERHEAD_RATIO,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this project experiment.")

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    configs = [
        FinalExperimentConfig(model_name="ResNet-152", batch_sizes=[2, 4, 8, 16]),
        FinalExperimentConfig(model_name="BERT-Base", batch_sizes=[1, 2, 4, 8]),
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {
        "ResNet-152": [],
        "BERT-Base": [],
    }
    started = time.time()

    for cfg in configs:
        for batch_size in cfg.batch_sizes:
            t0 = time.time()
            print(
                f"[run] {cfg.model_name} batch={batch_size} apply_ac=True "
                f"overhead_ratio={overhead_ratio}",
                flush=True,
            )
            row = _safe_one_setting(
                model_name=cfg.model_name,
                batch_size=batch_size,
                device=device,
                apply_ac=True,
                overhead_ratio=overhead_ratio,
            )
            elapsed = time.time() - t0
            no_ac_peak = row["baseline_profile"]["peak_cuda_mib"]
            ac_peak = row.get("ac_profile", row["baseline_profile"])["peak_cuda_mib"]
            no_ac_lat = row["baseline_iter_latency"]["mean_ms"]
            ac_lat = row.get("ac_iter_latency", row["baseline_iter_latency"])["mean_ms"]
            n_sel = row.get("selection", {}).get("n_selected", 0)
            print(
                f"[ok ] {cfg.model_name} batch={batch_size} "
                f"selected={n_sel} "
                f"peak {no_ac_peak:.1f}->{ac_peak:.1f} MiB "
                f"({(ac_peak - no_ac_peak) / max(no_ac_peak, 1e-9) * 100:+.1f}%), "
                f"latency {no_ac_lat:.1f}->{ac_lat:.1f} ms "
                f"({(ac_lat - no_ac_lat) / max(no_ac_lat, 1e-9) * 100:+.1f}%)  "
                f"[{elapsed:.1f}s]",
                flush=True,
            )
            all_results[cfg.model_name].append(row)

    return {
        "phase_1_completed": True,
        "phase_2_completed": True,
        "phase_3_completed": True,
        "wall_time_sec": time.time() - started,
        "overhead_ratio_used": overhead_ratio,
        "results": all_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overhead-ratio",
        type=float,
        default=DEFAULT_OVERHEAD_RATIO,
        help="Maximum recompute overhead as fraction of forward runtime.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = run_final_experiment(overhead_ratio=args.overhead_ratio)
    json_path = os.path.join(OUTPUT_DIR, "final_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {json_path}")

    rows = results["results"]
    _plot_peak_memory_with_without_ac(
        rows, os.path.join(OUTPUT_DIR, "final_peak_memory.png")
    )
    _plot_peak_breakdown_with_without_ac(
        rows, os.path.join(OUTPUT_DIR, "final_peak_breakdown.png")
    )
    _plot_iter_latency_with_without_ac(
        rows, os.path.join(OUTPUT_DIR, "final_iter_latency.png")
    )
    print(f"Wrote: {os.path.join(OUTPUT_DIR, 'final_peak_memory.png')}")
    print(f"Wrote: {os.path.join(OUTPUT_DIR, 'final_peak_breakdown.png')}")
    print(f"Wrote: {os.path.join(OUTPUT_DIR, 'final_iter_latency.png')}")
