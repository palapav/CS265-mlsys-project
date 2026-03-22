import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.models import resnet152
from transformers import BertConfig, BertForMaskedLM

from graph_prof import GraphProfiler, NodeType
from graph_tracer import SEPFunction, compile


@dataclass
class ExperimentConfig:
    model_name: str
    batch_sizes: List[int]


def _to_mib(nbytes: float) -> float:
    return float(nbytes) / (1024.0 * 1024.0)


def _init_optimizer_states(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param)
    optimizer.step()
    optimizer.zero_grad()


def _collect_optimizer_state_ptrs(optimizer: torch.optim.Optimizer) -> set[int]:
    ptrs: set[int] = set()
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                ptrs.add(value.data_ptr())
    return ptrs


def _infer_placeholder_node_types(
    gm: torch.fx.GraphModule,
    args: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[torch.fx.Node, NodeType]:
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    placeholder_types: Dict[torch.fx.Node, NodeType] = {}
    if len(placeholders) > len(args):
        return placeholder_types

    param_ptrs = {param.data_ptr() for param in model.parameters()}
    opt_state_ptrs = _collect_optimizer_state_ptrs(optimizer)

    for idx, placeholder in enumerate(placeholders):
        value = args[idx]
        if not isinstance(value, torch.Tensor):
            continue
        ptr = value.data_ptr()
        if ptr in param_ptrs:
            placeholder_types[placeholder] = NodeType.PARAM
        elif ptr in opt_state_ptrs:
            placeholder_types[placeholder] = NodeType.OPT_STATE
    return placeholder_types


def _trace_window(profiler: GraphProfiler, center_idx: int, radius: int = 4) -> List[Dict[str, Any]]:
    if center_idx < 0 or center_idx >= len(profiler.nodes):
        return []
    start = max(0, center_idx - radius)
    end = min(len(profiler.nodes), center_idx + radius + 1)
    trace: List[Dict[str, Any]] = []
    for idx in range(start, end):
        node = profiler.nodes[idx]
        trace.append(
            {
                "idx": idx,
                "name": node.name,
                "op": str(node.op),
                "target": str(node.target),
            }
        )
    return trace


def _run_profiled_iteration(
    train_step_fn: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    example_inputs: Any,
    warmup_iters: int = 1,
    profile_iters: int = 2,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    def graph_transformation(gm: torch.fx.GraphModule, args: Any) -> torch.fx.GraphModule:
        placeholder_node_types = _infer_placeholder_node_types(gm, args, model, optimizer)
        profiler = GraphProfiler(gm, placeholder_node_types=placeholder_node_types)
        with torch.no_grad():
            for _ in range(warmup_iters):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(profile_iters):
                profiler.run(*args)
        profiler.aggregate_stats()

        node_type_counts = Counter(profiler.node_types.values())
        num_placeholders = sum(1 for node in profiler.nodes if node.op == "placeholder")
        categorized_placeholders = (
            len(profiler.param_placeholders)
            + len(profiler.grad_placeholders)
            + len(profiler.opt_state_placeholders)
        )

        top_runtime_nodes = sorted(
            profiler.nodes, key=lambda n: profiler.avg_runtime_ms.get(n, 0.0), reverse=True
        )[:10]
        largest_activations = sorted(
            list(profiler.activation_nodes),
            key=lambda n: profiler.avg_output_bytes.get(n, 0),
            reverse=True,
        )[:10]

        summary.update(
            {
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
                "placeholder_role_counts": {
                    "param": len(profiler.param_placeholders),
                    "grad": len(profiler.grad_placeholders),
                    "opt_state": len(profiler.opt_state_placeholders),
                    "other": max(0, num_placeholders - categorized_placeholders),
                    "total": num_placeholders,
                },
                "peak_breakdown_live_mib": {
                    "param": _to_mib(
                        profiler.avg_peak_by_type_bytes.get(NodeType.PARAM, 0.0)
                    ),
                    "act": _to_mib(
                        profiler.avg_peak_by_type_bytes.get(NodeType.ACT, 0.0)
                    ),
                    "grad": _to_mib(
                        profiler.avg_peak_by_type_bytes.get(NodeType.GRAD, 0.0)
                    ),
                    "opt_state": _to_mib(
                        profiler.avg_peak_by_type_bytes.get(NodeType.OPT_STATE, 0.0)
                    ),
                    "other": _to_mib(
                        profiler.avg_peak_by_type_bytes.get(NodeType.OTHER, 0.0)
                    ),
                },
                "peak_category_max_live_mib": {
                    "param": _to_mib(
                        profiler.avg_max_by_type_bytes.get(NodeType.PARAM, 0.0)
                    ),
                    "act": _to_mib(
                        profiler.avg_max_by_type_bytes.get(NodeType.ACT, 0.0)
                    ),
                    "grad": _to_mib(
                        profiler.avg_max_by_type_bytes.get(NodeType.GRAD, 0.0)
                    ),
                    "opt_state": _to_mib(
                        profiler.avg_max_by_type_bytes.get(NodeType.OPT_STATE, 0.0)
                    ),
                    "other": _to_mib(
                        profiler.avg_max_by_type_bytes.get(NodeType.OTHER, 0.0)
                    ),
                },
                "graph_evidence": {
                    "separator_validation": {
                        "forward_separator_found": profiler.sep_node is not None,
                        "backward_separator_found": profiler.sep_backward_node is not None,
                        "forward_before_backward": profiler.forward_end_idx
                        < profiler.backward_start_idx,
                        "optimizer_after_backward": profiler.optimizer_start_idx
                        >= profiler.backward_start_idx,
                    },
                    "boundary_indices": {
                        "forward_end_idx": profiler.forward_end_idx,
                        "backward_start_idx": profiler.backward_start_idx,
                        "optimizer_start_idx": profiler.optimizer_start_idx,
                    },
                    "region_node_counts": {
                        "forward_region_nodes": profiler.forward_end_idx + 1,
                        "backward_compute_nodes": max(
                            0, profiler.optimizer_start_idx - profiler.backward_start_idx
                        ),
                        "optimizer_region_nodes": max(
                            0, len(profiler.nodes) - profiler.optimizer_start_idx
                        ),
                    },
                    "separator_trace_window": _trace_window(
                        profiler, profiler.backward_start_idx
                    ),
                    "optimizer_trace_window": _trace_window(
                        profiler, profiler.optimizer_start_idx
                    ),
                },
                "top_runtime_ops": [
                    {
                        "name": n.name,
                        "op": n.op,
                        "target": str(n.target),
                        "avg_runtime_ms": profiler.avg_runtime_ms.get(n, 0.0),
                        "output_mib": _to_mib(profiler.avg_output_bytes.get(n, 0.0)),
                    }
                    for n in top_runtime_nodes
                ],
                "largest_activations": [
                    {
                        "name": n.name,
                        "size_mib": _to_mib(profiler.avg_output_bytes.get(n, 0)),
                        "last_forward_use": profiler.last_forward_use[n].name,
                        "first_backward_use": profiler.first_backward_use[n].name,
                    }
                    for n in largest_activations
                ],
            }
        )
        return gm

    compiled_fn = compile(train_step_fn, graph_transformation)
    compiled_fn(model, optimizer, example_inputs)
    return summary


def _build_resnet152_inputs(
    batch_size: int, device: torch.device
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Any, Any]:
    model = resnet152(weights=None).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, foreach=True, capturable=True
    )
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    example_inputs = (inputs, labels)

    def train_step(model: torch.nn.Module, optim: torch.optim.Optimizer, ex_inputs: Any):
        logits = model(ex_inputs[0])
        loss = F.cross_entropy(logits, ex_inputs[1])
        loss = SEPFunction.apply(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    return model, optimizer, example_inputs, train_step


def _build_bert_inputs(
    batch_size: int, device: torch.device, seq_len: int = 512
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Any, Any]:
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
    )
    model = BertForMaskedLM(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, foreach=True, capturable=True
    )
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    example_inputs = {"input_ids": input_ids, "labels": labels}

    def train_step(model: torch.nn.Module, optim: torch.optim.Optimizer, ex_inputs: Any):
        outputs = model(**ex_inputs)
        loss = outputs.loss
        loss = SEPFunction.apply(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    return model, optimizer, example_inputs, train_step


def _plot_peak_memory(results: Dict[str, List[Dict[str, Any]]], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
    for ax_idx, model_name in enumerate(["ResNet-152", "BERT-Base"]):
        model_results = results[model_name]
        batch_sizes = [entry["batch_size"] for entry in model_results]
        peak_cuda = [entry["profile"]["peak_cuda_mib"] for entry in model_results]
        peak_live = [entry["profile"]["peak_live_mib"] for entry in model_results]

        x = list(range(len(batch_sizes)))
        bar_w = 0.4
        axes[ax_idx].bar(
            [i - bar_w / 2 for i in x], peak_cuda, width=bar_w, label="CUDA peak"
        )
        axes[ax_idx].bar(
            [i + bar_w / 2 for i in x], peak_live, width=bar_w, label="Live-tensor peak"
        )
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels([str(bs) for bs in batch_sizes])
        axes[ax_idx].set_xlabel("Mini-batch size")
        axes[ax_idx].set_ylabel("Peak memory (MiB)")
        axes[ax_idx].set_title(f"{model_name} (w/o AC)")
        axes[ax_idx].grid(alpha=0.3, axis="y")
        axes[ax_idx].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _plot_peak_breakdown_at_peak(
    results: Dict[str, List[Dict[str, Any]]], out_path: str
) -> None:
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
    model_order = ["ResNet-152", "BERT-Base"]
    baseline_entries: Dict[str, Dict[str, Any]] = {}
    for model_name in model_order:
        model_results = results.get(model_name, [])
        if not model_results:
            continue
        baseline_entries[model_name] = min(model_results, key=lambda x: x["batch_size"])

    if not baseline_entries:
        return

    labels = []
    totals = []
    for model_name in model_order:
        if model_name not in baseline_entries:
            continue
        entry = baseline_entries[model_name]
        labels.append(f"{model_name}\n(batch={entry['batch_size']})")
        totals.append(entry["profile"]["peak_live_mib"])

    x = list(range(len(labels)))
    bottoms = [0.0 for _ in labels]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=160)
    for key, pretty_name in categories:
        values = []
        for model_name in model_order:
            if model_name not in baseline_entries:
                continue
            values.append(
                baseline_entries[model_name]["profile"]["peak_breakdown_live_mib"].get(
                    key, 0.0
                )
            )
        ax.bar(
            x,
            values,
            width=0.6,
            bottom=bottoms,
            label=pretty_name,
            color=colors[key],
        )
        bottoms = [b + v for b, v in zip(bottoms, values)]

    for idx, total in enumerate(totals):
        ax.text(
            idx,
            total + max(totals) * 0.015,
            f"{total:.1f} MiB",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Live memory (MiB)")
    ax.set_title("Peak Live-Memory Breakdown at Total Peak (w/o AC)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(ncols=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def run_midway_checkin() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this project experiment.")

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    config = [
        ExperimentConfig(model_name="ResNet-152", batch_sizes=[2, 4, 8]),
        ExperimentConfig(model_name="BERT-Base", batch_sizes=[1, 2, 4, 8]),
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {"ResNet-152": [], "BERT-Base": []}

    for exp_cfg in config:
        for batch_size in exp_cfg.batch_sizes:
            if exp_cfg.model_name == "ResNet-152":
                model, optimizer, example_inputs, train_step = _build_resnet152_inputs(
                    batch_size=batch_size, device=device
                )
            else:
                model, optimizer, example_inputs, train_step = _build_bert_inputs(
                    batch_size=batch_size, device=device
                )

            _init_optimizer_states(model, optimizer)
            profile_summary = _run_profiled_iteration(
                train_step_fn=train_step,
                model=model,
                optimizer=optimizer,
                example_inputs=example_inputs,
            )
            all_results[exp_cfg.model_name].append(
                {"batch_size": batch_size, "profile": profile_summary}
            )
            torch.cuda.empty_cache()

    out = {
        "phase_1_completed": True,
        "deliverable_4a_without_ac": {
            "description": "Compute/memory profiling statistics and static activation analysis",
            "results": all_results,
        },
    }
    return out


if __name__ == "__main__":
    results = run_midway_checkin()
    json_path = "midway_results_wo_ac.json"
    plot_path = "midway_peak_memory_wo_ac.png"
    breakdown_plot_path = "midway_peak_breakdown_at_peak_wo_ac.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _plot_peak_memory(results["deliverable_4a_without_ac"]["results"], plot_path)
    _plot_peak_breakdown_at_peak(
        results["deliverable_4a_without_ac"]["results"], breakdown_plot_path
    )
    print(f"Wrote: {json_path}")
    print(f"Wrote: {plot_path}")
    print(f"Wrote: {breakdown_plot_path}")
