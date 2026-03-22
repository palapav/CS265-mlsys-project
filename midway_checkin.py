import json
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
        profiler = GraphProfiler(gm)
        with torch.no_grad():
            for _ in range(warmup_iters):
                profiler.run(*args)
            profiler.reset_stats()
            for _ in range(profile_iters):
                profiler.run(*args)
        profiler.aggregate_stats()

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
                    "other": _to_mib(
                        profiler.avg_peak_by_type_bytes.get(NodeType.OTHER, 0.0)
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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _plot_peak_memory(results["deliverable_4a_without_ac"]["results"], plot_path)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {plot_path}")
