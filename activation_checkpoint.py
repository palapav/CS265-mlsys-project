import torch
import torch.nn as nn
import torch.fx as fx
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from graph_tracer import SEPFunction


def _extract_subgraph(
    joint_graph: fx.Graph,
    inputs: List[fx.Node],
    outputs: List[fx.Node],
) -> fx.Graph:
    """Wrapper around `_extract_graph_with_inputs_outputs` that papers over
    the PyTorch >= 2.4 signature change which now requires an additional
    `outputs_descs` positional argument (one entry per output, used only as
    `out.meta["desc"]`)."""
    try:
        return _extract_graph_with_inputs_outputs(
            joint_graph=joint_graph,
            inputs=inputs,
            outputs=outputs,
            outputs_descs=[None] * len(outputs),
        )
    except TypeError:
        # Older PyTorch (< 2.4) does not accept outputs_descs.
        return _extract_graph_with_inputs_outputs(
            joint_graph=joint_graph,
            inputs=inputs,
            outputs=outputs,
        )


# We define a custom function that takes in two weight matrices that require
# gradients to be computed and an input data matrix. The function returns the
# gradients of the weight matrices with respect to the loss (sum in our
# example). NOTE: The custom function mimics a simple two layer liner neural
# network with relu activation functions and a sum loss function.
def custom_fn(w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    z = torch.mm(w1, x)
    z = nn.functional.relu(z)
    z = torch.mm(z, w2)
    z = nn.functional.relu(z)
    z = z.sum()
    z = SEPFunction.apply(z)
    z.backward()
    return w1.grad, w2.grad


def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
) -> None:
    old_node_users = old_node.users
    for node in reversed(graph.nodes):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def remove_detach_nodes(gm: fx.GraphModule) -> fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def get_name_to_node_map(gm: fx.GraphModule) -> Dict[str, fx.Node]:
    name_to_node = {}
    for node in gm.graph.nodes:
        name_to_node[node.name] = node
    return name_to_node


def activation_checkpointing(gm: fx.GraphModule) -> fx.GraphModule:
    # NOTE: You need to create the function for your project and call it inside
    # the graph_transformation function after performing graph profiling.

    # In this example we are going to recompute one of the relu activations for the
    # backward pass instead of saving it. We know from our custom function
    # that we have 2 intermeidate nodes: ['relu', 'relu_1']

    # So the intermediate node to recompute is: ['relu'] and
    # intermediate nodes to checkpoint (retain) are: ['relu_1']

    # Nodes required to recompute 'relu' are ['w1_1', 'x_1']
    # First back use is at node 't'

    # NOTE: For your project, you will use GraphProfiler to identify the
    # intermediate nodes, their first back access, last forward access and
    # then MuTWO's algorithm to select the intermediate 'nodes_to_recompute' and
    # checkpoint (retain). The 'nodes_required_to_recompute' any of the
    # intermediate nodes MUST be a subset of the placeholder nodes and the
    # intermediate nodes that are checkpointed.

    name_to_node = get_name_to_node_map(gm)
    first_back_access = name_to_node["t"]
    node_to_recompute = [name_to_node["relu"]]
    node_to_recompute_names = ["relu"]
    nodes_required_to_recompute = [name_to_node["w1_1"], name_to_node["x_1"]]

    # NOTE: we cannot directly use 'mm' to recompute 'relu' since 'mm' is not an
    # intermediate node that is retained (checkpointed).

    # Obtain a sub-graph that recomputes the required nodes
    recompute_subgraph = _extract_subgraph(
        joint_graph=gm.graph,
        inputs=nodes_required_to_recompute,
        outputs=node_to_recompute,
    )
    print("Extracted recomputation sub-graph: ")
    recompute_subgraph.print_tabular()

    # Insert the nodes of the new sub-graph in the old graph before the first
    # backward access of the node to be recomputed.
    with gm.graph.inserting_before(first_back_access):
        for n in recompute_subgraph.nodes:
            if n.op == "placeholder" or n.op == "output":
                continue
            # Copy the nodes of the new sub-graph to old graph and transform its
            # inputs to match the old-graph inputs. The arg_transform function
            # will pass the input arguments of the new node and will expect a
            # mapping to the nodes of the old graph.
            new_node = gm.graph.node_copy(
                n, arg_transform=lambda arg: name_to_node[arg.name]
            )

            if n.name in node_to_recompute_names:
                old_node = name_to_node[n.name]
                # Replace all the uses of the old node with new recomputation node
                replace_subsequent_uses_of(
                    gm.graph, old_node=old_node, new_node=new_node
                )
            # Add the new node to our name to node mapping
            name_to_node[n.name] = new_node

    gm.graph.lint()
    gm.recompile()
    return gm


# =============================================================================
# Phase 2 + Phase 3: generic mu-TWO-style selection and rewriter.
#
# These functions replace the hard-coded "drop relu" example above with an
# automated pipeline that consumes a populated GraphProfiler (Phase 1):
#
#   selected = select_recomputations(profiler, ...)
#   apply_activation_checkpointing(gm, profiler, selected)
#
# Both the selector and the rewriter follow the same recipe as the existing
# tutorial code (`_extract_graph_with_inputs_outputs` + `node_copy(arg_transform)`
# + `replace_subsequent_uses_of`). The novelty is:
#
# 1. The selector is greedy, iterative, and ratio-based: at each step it picks
#    the candidate activation with the largest "marginal new bytes saved per
#    millisecond of recompute overhead", where the recompute cost reflects the
#    *current* retained set (so dropping an early activation makes later
#    activations cheaper to drop, which the iterative rule captures).
# 2. The rewriter places each recomputation just before the earliest *future*
#    consumer of the activation in the modified graph, where consumers are
#    both original backward users and any later recompute that depends on it.
#    This lets recomputed activations chain (B's recompute can reuse A's
#    recompute) instead of forcing every drop to recompute from placeholders.
# =============================================================================


# Targets that mutate state and must never appear in a recompute subgraph.
# `copy_` is the optimizer write-back; `_foreach_*` mutating variants update
# placeholder tensors in place. We never select these as candidates and we
# refuse to recompute through them.
_UNSAFE_TARGET_SUBSTRINGS: Tuple[str, ...] = (
    "copy_",
    "set_",
    "_foreach_add_",
    "_foreach_sub_",
    "_foreach_mul_",
    "_foreach_div_",
    "_foreach_addcmul_",
    "_foreach_addcdiv_",
)

# Targets that are pure aliases of an input tensor and so contribute zero
# marginal storage. We still allow these as recompute *intermediates* (they
# show up in subgraphs and are cheap to re-execute), but we never pick them as
# the activation we drop -- dropping a view doesn't free anything.
_ALIAS_TARGET_SUBSTRINGS: Tuple[str, ...] = (
    "aten.view",
    "aten.t.",
    "aten.transpose",
    "aten.permute",
    "aten.reshape",
    "aten.expand",
    "aten.as_strided",
    "aten.unsqueeze",
    "aten.squeeze",
    "aten.flatten",
    "aten.detach",
    "aten.alias",
    "separator.sep",
)


def _target_str(node: fx.Node) -> str:
    return str(node.target)


def _is_unsafe_target(node: fx.Node) -> bool:
    target = _target_str(node)
    return any(s in target for s in _UNSAFE_TARGET_SUBSTRINGS)


def _is_alias_target(node: fx.Node) -> bool:
    target = _target_str(node)
    return any(s in target for s in _ALIAS_TARGET_SUBSTRINGS)


def _is_safe_to_recompute(node: fx.Node, profiler: Any) -> bool:
    """A node is a safe AC *candidate* (can be dropped + recomputed) if:
      - it's a strict activation (forward-produced + has direct backward user),
      - its forward op is not stateful (no `copy_`, no in-place foreach update),
      - it actually allocates new storage (marginal_bytes > 0), so dropping it
        can free memory rather than just deleting an alias view.
    The middle of a recompute subgraph may still contain alias ops; we only
    refuse to *select* aliases as the dropped activation.
    """
    if node.op != "call_function":
        return False
    if _is_unsafe_target(node):
        return False
    if _is_alias_target(node):
        return False
    if profiler.avg_new_storage_bytes.get(node, 0) <= 0:
        return False
    return True


def _is_recomputable_op(node: fx.Node) -> bool:
    """A node is allowed to appear as an *intermediate* in a recompute subgraph
    if it doesn't have an in-place mutation effect."""
    if node.op != "call_function":
        return False
    if _is_unsafe_target(node):
        return False
    return True


def _forward_ancestors_to_retained(
    node: fx.Node,
    retained: Set[fx.Node],
    profiler: Any,
) -> Set[fx.Node]:
    """Set of forward-region, non-placeholder, non-retained ancestors of `node`
    needed to recompute it from `retained`. Walks `all_input_nodes` upwards
    until it hits a retained boundary or a placeholder/get_attr/output.
    """
    visited: Set[fx.Node] = set()
    out: Set[fx.Node] = set()
    stack: List[fx.Node] = [node]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        if cur in retained:
            continue
        if cur.op in ("placeholder", "get_attr", "output"):
            continue
        if not profiler._is_forward_region(cur):
            # Recompute subgraph must stay in forward region; if we hit a
            # backward op while walking inputs that means our retained set is
            # insufficient, but we still bail out instead of pulling backward
            # nodes into the recompute.
            continue
        out.add(cur)
        stack.extend(cur.all_input_nodes)
    return out


def _recompute_cost_ms(profiler: Any, ancestors: Iterable[fx.Node]) -> float:
    return float(sum(profiler.avg_runtime_ms.get(n, 0.0) for n in ancestors))


def _forward_runtime_ms(profiler: Any) -> float:
    return float(
        sum(
            profiler.avg_runtime_ms.get(n, 0.0)
            for n in profiler.nodes
            if profiler._is_forward_region(n)
        )
    )


def select_recomputations(
    profiler: Any,
    max_recompute_overhead_ratio: float = 0.5,
    target_memory_savings_mib: Optional[float] = None,
    min_marginal_mib: float = 0.5,
    lifetime_weight: float = 1.0,
) -> Dict[str, Any]:
    """Greedy mu-TWO-style selection of activations to drop and recompute.

    Each candidate activation is scored independently: its recompute cost is
    the time needed to reproduce it from `placeholders + every other
    activation` (i.e. assuming all other activations are still retained).
    This corresponds directly to the rewriter's *independent* recompute
    blocks -- each dropped activation gets its own self-contained recompute
    subgraph at its first backward use, so its intermediate tensors live
    only inside that block and don't pile up across drops.

    The score for a candidate is:

        score = (marginal_bytes * lifetime_factor) / recompute_cost_ms

    where `lifetime_factor = lifetime_nodes ** lifetime_weight`. The
    `lifetime_nodes` is the gap between the activation's last forward use
    and its first backward use, in graph node indices -- a proxy for "how
    much wall-clock time this tensor sits on memory waiting to be used".
    Activations with small marginal storage (under `min_marginal_mib`) are
    dropped from the candidate pool entirely; otherwise the budget can be
    eaten by tiny activations that don't materially shrink peak memory.

    Greedy admission proceeds until either:
      - the cumulative recompute cost would exceed the overhead budget
        (`max_recompute_overhead_ratio * forward_runtime_ms`), or
      - the cumulative bytes saved exceeds `target_memory_savings_mib`.
    """
    placeholder_set: Set[fx.Node] = set(
        n for n in profiler.nodes if n.op == "placeholder"
    )
    all_acts: Set[fx.Node] = set(profiler.activation_nodes)
    safe_acts = [a for a in all_acts if _is_safe_to_recompute(a, profiler)]

    target_bytes = (
        int(target_memory_savings_mib * 1024 * 1024)
        if target_memory_savings_mib is not None
        else None
    )
    overhead_budget_ms = max_recompute_overhead_ratio * max(
        _forward_runtime_ms(profiler), 1e-6
    )
    min_marginal_bytes = int(min_marginal_mib * 1024 * 1024)

    # Pre-filter for size and unsafe chains. We compute the lifetime once;
    # cost is recomputed on the fly inside the iterative loop.
    skipped_unsafe_chain: List[fx.Node] = []
    skipped_too_small: List[fx.Node] = []
    candidate_meta: Dict[fx.Node, Tuple[int, int]] = {}  # cand -> (bytes, lifetime)
    for cand in safe_acts:
        bytes_saved = int(profiler.avg_new_storage_bytes.get(cand, 0))
        if bytes_saved < min_marginal_bytes:
            skipped_too_small.append(cand)
            continue
        last_fwd = profiler.last_forward_use[cand]
        first_bwd = profiler.first_backward_use[cand]
        lifetime_nodes = max(
            1,
            profiler.node_to_idx[first_bwd] - profiler.node_to_idx[last_fwd],
        )
        candidate_meta[cand] = (bytes_saved, lifetime_nodes)

    selected: List[fx.Node] = []
    selected_set: Set[fx.Node] = set()
    total_recompute_ms = 0.0
    total_bytes_saved = 0

    # Iterative greedy: at each step we re-rank every remaining candidate
    # using the *current* `selected_set`, because the rewriter's recompute
    # block for cand will walk back through every other selected activation
    # (they sit in selected_set, not in `retained`). This makes the cost
    # estimate match the rewriter's actual recompute path -- in particular
    # it correctly attributes the cost of an expensive shared op
    # (e.g. a vocab-projection addmm in BERT) to *every* candidate whose
    # chain crosses it after we drop the in-between activations.
    candidates: Set[fx.Node] = set(candidate_meta.keys())
    while candidates:
        retained = placeholder_set | (all_acts - selected_set)
        best: Optional[Tuple[fx.Node, int, float, float, int]] = None
        unsafe_now: List[fx.Node] = []
        for cand in candidates:
            ancestors = _forward_ancestors_to_retained(cand, retained, profiler)
            if any(
                a.op == "call_function" and _is_unsafe_target(a)
                for a in ancestors
            ):
                unsafe_now.append(cand)
                continue
            cost_ms = max(_recompute_cost_ms(profiler, ancestors), 1e-3)
            bytes_saved, lifetime_nodes = candidate_meta[cand]
            lifetime_factor = lifetime_nodes ** lifetime_weight
            score = (bytes_saved * lifetime_factor) / cost_ms
            if best is None or score > best[3]:
                best = (cand, bytes_saved, cost_ms, score, lifetime_nodes)

        for c in unsafe_now:
            candidates.discard(c)
            skipped_unsafe_chain.append(c)
        if best is None:
            break
        cand, bytes_saved, cost_ms, score, lifetime_nodes = best
        if total_recompute_ms + cost_ms > overhead_budget_ms:
            # Even the highest-score remaining candidate would blow the
            # budget; admitting any other (lower-score) candidate also
            # only makes things worse, so stop.
            break
        selected.append(cand)
        selected_set.add(cand)
        candidates.discard(cand)
        total_recompute_ms += cost_ms
        total_bytes_saved += bytes_saved
        if target_bytes is not None and total_bytes_saved >= target_bytes:
            break

    return {
        "selected": selected,
        "total_bytes_saved": total_bytes_saved,
        "total_bytes_saved_mib": total_bytes_saved / (1024.0 * 1024.0),
        "total_recompute_ms": total_recompute_ms,
        "overhead_budget_ms": overhead_budget_ms,
        "forward_runtime_ms": _forward_runtime_ms(profiler),
        "n_safe_candidates": len(safe_acts),
        "n_skipped_unsafe_chain": len(skipped_unsafe_chain),
        "n_skipped_too_small": len(skipped_too_small),
    }


def _compute_anchors(
    selected: Sequence[fx.Node], profiler: Any
) -> Dict[fx.Node, fx.Node]:
    """Each selected activation gets an anchor equal to its first backward
    use. Recompute blocks are independent (each block only touches its own
    fresh intermediate copies), so no cross-anchor propagation is required
    and lifetimes stay short."""
    return {a: profiler.first_backward_use[a] for a in selected}


def apply_activation_checkpointing(
    gm: fx.GraphModule,
    profiler: Any,
    selected: Sequence[fx.Node],
) -> fx.GraphModule:
    """Phase 3 graph rewriter.

    For each activation in `selected`, drop its forward result from being
    saved across to backward by inserting a fresh recompute subgraph just
    before its first consumer (in the modified graph) and rerouting backward
    consumers to the recomputed copy. Dependent recomputations chain so a
    later recompute reuses the earlier one's recomputed copy rather than
    re-walking back to placeholders.
    """
    if not selected:
        gm.graph.lint()
        gm.recompile()
        return gm

    name_to_node: Dict[str, fx.Node] = {n.name: n for n in gm.graph.nodes}
    selected_set: Set[fx.Node] = set(selected)
    placeholder_set: Set[fx.Node] = set(
        n for n in profiler.nodes if n.op == "placeholder"
    )
    retained_act = set(profiler.activation_nodes) - selected_set
    base_retained: Set[fx.Node] = placeholder_set | retained_act

    anchor = _compute_anchors(selected, profiler)

    # Process in (anchor.idx ascending, then forward_idx ascending). Each
    # block is independent of the others, so this order doesn't affect
    # correctness; it just keeps inserts moving forward through the graph.
    insertion_order = sorted(
        selected,
        key=lambda n: (profiler.node_to_idx[anchor[n]], profiler.node_to_idx[n]),
    )

    for act in insertion_order:
        # Boundary for this activation's recompute is just the base
        # retained set: placeholders + activations we did not select.
        # Other selected activations are walked *through*, so each block
        # gets its own fresh copies of any selected ancestors. That keeps
        # intermediate lifetimes confined to this block instead of letting
        # them pile up in backward.
        ancestors = _forward_ancestors_to_retained(act, base_retained, profiler)
        # Refuse to recompute through ops with side effects (mutating
        # foreach updates, copy_ writes that the partitioner would
        # otherwise pull in via dead-code preservation, etc.).
        for n in ancestors:
            if not _is_recomputable_op(n):
                raise RuntimeError(
                    f"Refusing to recompute through unsafe op {n.target} "
                    f"while rewriting activation {act.name}"
                )

        # `local_map` is a per-iteration extension of `name_to_node`. It
        # binds each freshly-copied intermediate by its original name so
        # arg_transform inside this iteration resolves to the fresh copy,
        # without leaking that mapping to later iterations (which would
        # incorrectly chain the next recompute through this one's
        # intermediates and prolong their lifetimes).
        local_map: Dict[str, fx.Node] = dict(name_to_node)

        topo_list = sorted(
            ancestors | {act}, key=lambda n: profiler.node_to_idx[n]
        )
        anchor_node = anchor[act]
        recomputed_act: Optional[fx.Node] = None
        with gm.graph.inserting_before(anchor_node):
            for n in topo_list:
                new_node = gm.graph.node_copy(
                    n, arg_transform=lambda arg: local_map[arg.name]
                )
                if n is act:
                    recomputed_act = new_node
                else:
                    local_map[n.name] = new_node

        assert recomputed_act is not None
        # Reroute every *subsequent* user (backward consumers and later
        # recompute subgraphs that take `act` as a boundary input) to use
        # the recomputed copy. The original forward `act` node still
        # feeds its forward users, so we don't break the forward chain
        # that produced downstream forward activations.
        replace_subsequent_uses_of(
            gm.graph, old_node=act, new_node=recomputed_act
        )
        name_to_node[act.name] = recomputed_act

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


# =============================================================================


if __name__ == "__main__":
    # Create two weight matrices that require gradients and one input data matrix
    w1 = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    w2 = torch.randn(2048, 512, device="cuda", requires_grad=True)
    x = torch.randn(1024, 2048, device="cuda")

    # Create a graph module by tracing the the custom function with the given inputs
    graph_module = make_fx(custom_fn)(w1, w2, x)
    graph_module = remove_detach_nodes(graph_module)
    print("Original graph of custom fn (fwd+bwd): ")
    graph_module.graph.print_tabular()

    # Obtain the gradients of (w1, w2) using x as input to the traced function
    # NOTE: We have already captured the backward operations during tracing
    # hence we are executing in no grad mode
    with torch.no_grad():
        old_grads = graph_module(w1, w2, x)

    # Apply the activation checkpointing algorithm (check new node 'relu_2')
    new_graph_module = activation_checkpointing(graph_module)
    print("Modified graph of custom fn (fwd+bwd+activation_checkpointing): ")
    new_graph_module.graph.print_tabular()

    # Obtain the gradients of (w1, w2) using x as input to the activation
    # checkpointed function to recalculate them
    with torch.no_grad():
        new_grads = new_graph_module(w1, w2, x)

    # Verify that gradients produced with activation checkpointing equal the
    # ones obtained earlier with no optimization.
    print("Result verification")
    for old_grad, new_grad in zip(old_grads, new_grads):
        print(torch.allclose(old_grad, new_grad))
