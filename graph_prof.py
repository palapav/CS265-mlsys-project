from enum import Enum
import time
from typing import Any, Dict, Iterable, List, Set

import torch
import torch.fx as fx


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OPT_STATE = 3
    OTHER = 4


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        garbage_collect_values: bool = True,
        placeholder_node_types: Dict[fx.Node, NodeType] | None = None,
    ):
        super().__init__(module, garbage_collect_values)
        self.nodes: List[fx.Node] = list(self.module.graph.nodes)
        self.node_to_idx: Dict[fx.Node, int] = {
            node: idx for idx, node in enumerate(self.nodes)
        }
        self.placeholder_node_types = placeholder_node_types or {}

        self.sep_node = self._find_separator_node(backward=False)
        self.sep_backward_node = self._find_separator_node(backward=True)
        self.forward_end_idx = (
            self.node_to_idx[self.sep_node]
            if self.sep_node is not None
            else len(self.nodes) - 1
        )
        self.backward_start_idx = (
            self.node_to_idx[self.sep_backward_node]
            if self.sep_backward_node is not None
            else len(self.nodes)
        )
        self.optimizer_start_idx = self._find_optimizer_start_idx()

        self.optimizer_nodes = self._find_optimizer_nodes()
        self.param_nodes: Set[fx.Node] = set()
        self.grad_nodes: Set[fx.Node] = set()
        self.param_placeholders: Set[fx.Node] = set()
        self.grad_placeholders: Set[fx.Node] = set()
        self.opt_state_placeholders: Set[fx.Node] = set()

        if self.placeholder_node_types:
            for placeholder, node_type in self.placeholder_node_types.items():
                if placeholder.op != OP.PLACEHOLDER:
                    continue
                if node_type == NodeType.PARAM:
                    self.param_placeholders.add(placeholder)
                elif node_type == NodeType.GRAD:
                    self.grad_placeholders.add(placeholder)
                elif node_type == NodeType.OPT_STATE:
                    self.opt_state_placeholders.add(placeholder)
        else:
            for optimizer_node in self.optimizer_nodes:
                if len(optimizer_node.args) > 0:
                    self.param_nodes.update(self._collect_nodes(optimizer_node.args[0]))
                if len(optimizer_node.args) > 1:
                    self.grad_nodes.update(self._collect_nodes(optimizer_node.args[1]))

            self.param_placeholders = self._resolve_to_placeholders(self.param_nodes)
            self.grad_placeholders = self._resolve_to_placeholders(self.grad_nodes)
            if not self.param_placeholders:
                # Fallback path for decomposed optimizer graphs where fused_adam is
                # lowered into foreach/copy ops and explicit param lists are absent.
                self.param_placeholders = self._infer_param_placeholders_from_mutations()

        self.grad_related_nodes = self._collect_ancestors(self.grad_nodes)

        self.optimizer_state_update_nodes: Set[fx.Node] = set()
        self.param_update_nodes: Set[fx.Node] = set()
        optimizer_update_src_for_state: Set[fx.Node] = set()
        optimizer_update_src_for_param: Set[fx.Node] = set()
        for node in self.nodes:
            if not self._is_optimizer_region(node):
                continue
            if node.op != OP.CALL_FUNCTION or "copy_" not in self._target_str(node):
                continue
            if len(node.all_input_nodes) < 2:
                continue
            dst = node.all_input_nodes[0]
            src = node.all_input_nodes[1]
            if dst in self.opt_state_placeholders:
                self.optimizer_state_update_nodes.add(node)
                optimizer_update_src_for_state.add(src)
            elif dst in self.param_placeholders:
                self.param_update_nodes.add(node)
                optimizer_update_src_for_param.add(src)
        self.optimizer_state_update_ancestors = self._collect_ancestors(
            optimizer_update_src_for_state
        )
        self.param_update_ancestors = self._collect_ancestors(optimizer_update_src_for_param)

        self.activation_nodes: Set[fx.Node] = set()
        self.last_forward_use: Dict[fx.Node, fx.Node] = {}
        self.first_backward_use: Dict[fx.Node, fx.Node] = {}
        self._analyze_activations()

        self.node_types: Dict[fx.Node, NodeType] = {}
        for node in self.nodes:
            self.node_types[node] = self._classify_node(node)

        self._template_use_count: Dict[fx.Node, int] = {
            node: len(node.users) for node in self.nodes
        }

        self.reset_stats()

    def _target_str(self, node: fx.Node) -> str:
        return str(node.target)

    def _find_separator_node(self, backward: bool) -> fx.Node | None:
        target_name = "separator.sep_backward.default" if backward else "separator.sep.default"
        for node in self.nodes:
            if node.op != OP.CALL_FUNCTION:
                continue
            if target_name in self._target_str(node):
                return node
        return None

    def _find_optimizer_nodes(self) -> List[fx.Node]:
        optimizer_nodes: List[fx.Node] = []
        for node in self.nodes:
            if node.op != OP.CALL_FUNCTION:
                continue
            if self.node_to_idx[node] < self.backward_start_idx:
                continue
            target = self._target_str(node)
            if "fused_adam" in target or "_foreach" in target or "copy_" in target:
                optimizer_nodes.append(node)
        return optimizer_nodes

    def _find_optimizer_start_idx(self) -> int:
        for node in self.nodes:
            if self.node_to_idx[node] < self.backward_start_idx:
                continue
            if node.op != OP.CALL_FUNCTION:
                continue
            target = self._target_str(node)
            if "fused_adam" in target or "_foreach" in target or "copy_" in target:
                return self.node_to_idx[node]
        return len(self.nodes)

    def _collect_nodes(self, arg: Any) -> Set[fx.Node]:
        nodes: Set[fx.Node] = set()
        if isinstance(arg, fx.Node):
            nodes.add(arg)
            return nodes
        if isinstance(arg, (list, tuple)):
            for item in arg:
                nodes.update(self._collect_nodes(item))
            return nodes
        if isinstance(arg, dict):
            for value in arg.values():
                nodes.update(self._collect_nodes(value))
            return nodes
        return nodes

    def _resolve_to_placeholders(self, nodes: Iterable[fx.Node]) -> Set[fx.Node]:
        placeholders: Set[fx.Node] = set()
        queue: List[fx.Node] = list(nodes)
        visited: Set[fx.Node] = set()
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur.op == OP.PLACEHOLDER:
                placeholders.add(cur)
                continue
            queue.extend(cur.all_input_nodes)
        return placeholders

    def _collect_ancestors(self, nodes: Iterable[fx.Node]) -> Set[fx.Node]:
        ancestors: Set[fx.Node] = set()
        queue: List[fx.Node] = list(nodes)
        while queue:
            cur = queue.pop()
            if cur in ancestors:
                continue
            ancestors.add(cur)
            queue.extend(cur.all_input_nodes)
        return ancestors

    def _infer_param_placeholders_from_mutations(self) -> Set[fx.Node]:
        mutable_placeholders: Set[fx.Node] = set()
        for node in self.nodes:
            if node.op != OP.CALL_FUNCTION:
                continue
            if "copy_" not in self._target_str(node):
                continue
            if not node.all_input_nodes:
                continue
            dst = node.all_input_nodes[0]
            if dst.op == OP.PLACEHOLDER:
                mutable_placeholders.add(dst)

        inferred_params: Set[fx.Node] = set()
        for placeholder in mutable_placeholders:
            if any(self._is_forward_region(user) for user in placeholder.users):
                inferred_params.add(placeholder)
        return inferred_params

    def _is_forward_region(self, node: fx.Node) -> bool:
        return self.node_to_idx[node] <= self.forward_end_idx

    def _is_backward_region(self, node: fx.Node) -> bool:
        return self.node_to_idx[node] >= self.backward_start_idx

    def _is_backward_compute_region(self, node: fx.Node) -> bool:
        idx = self.node_to_idx[node]
        return self.backward_start_idx <= idx < self.optimizer_start_idx

    def _is_optimizer_region(self, node: fx.Node) -> bool:
        return self.node_to_idx[node] >= self.optimizer_start_idx

    def _is_tensor_producing(self, node: fx.Node) -> bool:
        return node.op not in [OP.OUTPUT]

    def _analyze_activations(self) -> None:
        for node in self.nodes:
            if not self._is_forward_region(node):
                continue
            if node.op in [OP.PLACEHOLDER, OP.GET_ATTR, OP.OUTPUT]:
                continue
            if not self._is_tensor_producing(node):
                continue
            if self.sep_node is not None and node is self.sep_node:
                continue

            users = list(node.users.keys())
            backward_users = [u for u in users if self._is_backward_region(u)]
            if not backward_users:
                continue

            forward_users = [u for u in users if self._is_forward_region(u)]
            self.activation_nodes.add(node)

            if forward_users:
                last_fwd = max(forward_users, key=lambda n: self.node_to_idx[n])
            else:
                last_fwd = node
            first_bwd = min(backward_users, key=lambda n: self.node_to_idx[n])
            self.last_forward_use[node] = last_fwd
            self.first_backward_use[node] = first_bwd

    def _classify_node(self, node: fx.Node) -> NodeType:
        # Placeholder roles are determined by runtime tensor identity.
        if node.op == OP.PLACEHOLDER and node in self.param_placeholders:
            return NodeType.PARAM
        if node.op == OP.PLACEHOLDER and node in self.opt_state_placeholders:
            return NodeType.OPT_STATE
        if node.op == OP.PLACEHOLDER and node in self.grad_placeholders:
            return NodeType.GRAD
        if node in self.param_nodes:
            return NodeType.PARAM
        # Optimizer step write-backs.
        if node in self.optimizer_state_update_nodes:
            return NodeType.OPT_STATE
        if node in self.param_update_nodes:
            return NodeType.PARAM
        # Forward-region tensor-producing nodes (excluding placeholders, the
        # separator, and metadata-only ops) are categorized as ACT for memory
        # accounting. This covers both the strict "saved-for-backward"
        # activations (in `self.activation_nodes`, used by the Phase 2 AC
        # algorithm) and the forward-only transient buffers that are alive
        # only briefly during forward. The latter are not AC candidates, but
        # they are forward-pass activation working memory in the broad ML
        # sense, and lumping them into OTHER misleads the breakdown.
        if (
            self._is_forward_region(node)
            and node.op not in (OP.PLACEHOLDER, OP.GET_ATTR, OP.OUTPUT)
            and node is not self.sep_node
        ):
            return NodeType.ACT
        if self._is_backward_compute_region(node):
            return NodeType.GRAD
        if node in self.grad_related_nodes and self._is_backward_region(node):
            return NodeType.GRAD
        # Optimizer-region intermediates (foreach math, getitem unpacks of
        # _foreach_* lists, denom/sqrt buffers, addcdiv step intermediates,
        # etc.) are the optimizer step's working set. We deliberately do NOT
        # route param-update ancestors into PARAM: doing so misattributes Adam
        # working buffers (denom, v_hat, step temporaries) to model parameters
        # and inflates the PARAM bar by 2-3x at small batches when peak
        # occurs in the optimizer region.
        if self._is_optimizer_region(node):
            return NodeType.OPT_STATE
        return NodeType.OTHER

    def _extract_storages(self, value: Any) -> Dict[int, int]:
        """Walk a node-output value tree and return a dict mapping each unique
        underlying CUDA storage pointer to its byte size.

        We key by ``untyped_storage().data_ptr()`` (the start of the storage)
        rather than ``tensor.data_ptr()`` (which is the start of a possibly-
        offset view) so that all aliases (`view`, `t`, `transpose`,
        `as_strided`, `getitem` of a `_foreach_*` list, `copy_` destination,
        etc.) collapse to the same key. ``untyped_storage().nbytes()`` gives the
        size of the actual storage allocation, which is the right quantity for
        a memory model.
        """
        storages: Dict[int, int] = {}
        seen_tensor_ids: Set[int] = set()

        def _inner(v: Any) -> None:
            if isinstance(v, torch.Tensor):
                if id(v) in seen_tensor_ids:
                    return
                seen_tensor_ids.add(id(v))
                try:
                    storage = v.untyped_storage()
                    ptr = storage.data_ptr()
                    if ptr == 0:
                        return
                    if ptr not in storages:
                        storages[ptr] = int(storage.nbytes())
                except Exception:
                    ptr = v.data_ptr()
                    if ptr == 0:
                        return
                    if ptr not in storages:
                        storages[ptr] = int(v.nelement() * v.element_size())
            elif isinstance(v, (list, tuple)):
                for x in v:
                    _inner(x)
            elif isinstance(v, dict):
                for x in v.values():
                    _inner(x)

        _inner(value)
        return storages

    def _bytes_to_mib(self, nbytes: float) -> float:
        return nbytes / (1024.0 * 1024.0)

    def _is_inplace_update_node(self, node: fx.Node) -> bool:
        if node.op not in [OP.CALL_FUNCTION, OP.CALL_METHOD]:
            return False
        return "copy_" in self._target_str(node)

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> Any:
        # Storage-based liveness tracking. We refcount each unique CUDA storage
        # by the number of live fx.Nodes still holding a reference to it. This
        # gives a correct lower-bound view of memory: aliases never double-count
        # and in-place ops (copy_, foreach_*_) do not falsely add bytes.
        self._remaining_uses: Dict[fx.Node, int] = dict(self._template_use_count)
        self._node_storage_ptrs: Dict[fx.Node, Set[int]] = {}
        self._storage_bytes: Dict[int, int] = {}
        self._storage_refcount: Dict[int, int] = {}
        self._storage_category: Dict[int, NodeType] = {}
        self._cur_peak_total_bytes = 0
        self._cur_peak_total_composition_by_type_bytes: Dict[NodeType, int] = {
            node_type: 0 for node_type in NodeType
        }
        self._cur_peak_by_type_bytes: Dict[NodeType, int] = {
            node_type: 0 for node_type in NodeType
        }
        self._cur_torch_peak_bytes = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        result = super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

        self.peak_total_bytes_per_run.append(self._cur_peak_total_bytes)
        for node_type in NodeType:
            self.peak_total_composition_by_type_bytes_per_run[node_type].append(
                self._cur_peak_total_composition_by_type_bytes[node_type]
            )
            self.peak_by_type_bytes_per_run[node_type].append(
                self._cur_peak_by_type_bytes[node_type]
            )
        self.torch_peak_bytes_per_run.append(self._cur_torch_peak_bytes)
        return result

    def run_node(self, n: fx.Node) -> Any:
        use_cuda_timer = torch.cuda.is_available()
        if use_cuda_timer:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            before_alloc = torch.cuda.memory_allocated()
            start_event.record()
        else:
            before_alloc = 0
            start_time = time.perf_counter()

        result = super().run_node(n)

        if use_cuda_timer:
            end_event.record()
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            after_alloc = torch.cuda.memory_allocated()
            mem_delta_bytes = after_alloc - before_alloc
            self._cur_torch_peak_bytes = max(
                self._cur_torch_peak_bytes, torch.cuda.max_memory_allocated()
            )
        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            mem_delta_bytes = 0

        self.node_runtime_ms[n].append(elapsed_ms)
        self.node_memory_delta_bytes[n].append(mem_delta_bytes)

        # Increment refcounts on the unique storages this node's output holds.
        # New storages are attributed to this node's category; pre-existing
        # storages keep the category set by their first producer (typically a
        # placeholder, which is always processed before any aliasing op).
        node_output_bytes_total = 0
        node_new_storage_bytes = 0
        if self._is_tensor_producing(n):
            output_storages = self._extract_storages(result)
            if output_storages:
                node_ptrs: Set[int] = set(output_storages.keys())
                node_category = self.node_types.get(n, NodeType.OTHER)
                for ptr, size in output_storages.items():
                    node_output_bytes_total += size
                    if ptr not in self._storage_bytes:
                        self._storage_bytes[ptr] = size
                        self._storage_refcount[ptr] = 0
                        self._storage_category[ptr] = node_category
                        node_new_storage_bytes += size
                    self._storage_refcount[ptr] += 1
                self._node_storage_ptrs[n] = node_ptrs
        self.node_output_bytes[n] = max(
            self.node_output_bytes.get(n, 0), node_output_bytes_total
        )
        # Track marginal bytes (new storage allocated by this node). Aliasing
        # nodes (view/t/transpose/getitem of foreach lists, etc.) report 0
        # here, which is what the Phase 2 selection algorithm needs to avoid
        # "saving" memory by dropping a view of a tensor whose storage we
        # would still keep alive via other consumers.
        self.node_new_storage_bytes[n] = max(
            self.node_new_storage_bytes.get(n, 0), node_new_storage_bytes
        )

        # Decrement refcounts as inputs reach end-of-life.
        for input_node in n.all_input_nodes:
            if input_node not in self._remaining_uses:
                continue
            self._remaining_uses[input_node] -= 1
            if self._remaining_uses[input_node] <= 0:
                self._remaining_uses.pop(input_node, None)
                ptrs = self._node_storage_ptrs.pop(input_node, None)
                if ptrs:
                    for ptr in ptrs:
                        if ptr in self._storage_refcount:
                            self._storage_refcount[ptr] -= 1
                            if self._storage_refcount[ptr] <= 0:
                                self._storage_bytes.pop(ptr, None)
                                self._storage_refcount.pop(ptr, None)
                                self._storage_category.pop(ptr, None)

        live_by_type_bytes: Dict[NodeType, int] = {node_type: 0 for node_type in NodeType}
        for ptr, size in self._storage_bytes.items():
            cat = self._storage_category.get(ptr, NodeType.OTHER)
            live_by_type_bytes[cat] += size
        live_total_bytes = sum(self._storage_bytes.values())
        if live_total_bytes > self._cur_peak_total_bytes:
            self._cur_peak_total_bytes = live_total_bytes
            self._cur_peak_total_composition_by_type_bytes = dict(live_by_type_bytes)
        for node_type in NodeType:
            if live_by_type_bytes[node_type] > self._cur_peak_by_type_bytes[node_type]:
                self._cur_peak_by_type_bytes[node_type] = live_by_type_bytes[node_type]

        return result

    def aggregate_stats(self) -> None:
        # You are expected run the profiler for x warm-up iterations and y
        # actual measurement iterations. The run-time measurement then needs to
        # be averaged over the y runs.
        self.avg_runtime_ms: Dict[fx.Node, float] = {}
        self.avg_memory_delta_bytes: Dict[fx.Node, float] = {}
        self.avg_output_bytes: Dict[fx.Node, int] = dict(self.node_output_bytes)
        # Marginal new-storage bytes per node, used by Phase 2 selection so
        # alias-only nodes (views, transposes, getitem) are not credited with
        # savings they wouldn't actually deliver.
        self.avg_new_storage_bytes: Dict[fx.Node, int] = dict(
            self.node_new_storage_bytes
        )
        for node in self.nodes:
            runtime_samples = self.node_runtime_ms.get(node, [])
            mem_samples = self.node_memory_delta_bytes.get(node, [])
            self.avg_runtime_ms[node] = (
                sum(runtime_samples) / len(runtime_samples) if runtime_samples else 0.0
            )
            self.avg_memory_delta_bytes[node] = (
                sum(mem_samples) / len(mem_samples) if mem_samples else 0.0
            )

        self.avg_peak_total_bytes = (
            sum(self.peak_total_bytes_per_run) / len(self.peak_total_bytes_per_run)
            if self.peak_total_bytes_per_run
            else 0.0
        )
        self.avg_peak_by_type_bytes: Dict[NodeType, float] = {}
        self.avg_max_by_type_bytes: Dict[NodeType, float] = {}
        for node_type in NodeType:
            composition_samples = self.peak_total_composition_by_type_bytes_per_run[node_type]
            max_samples = self.peak_by_type_bytes_per_run[node_type]
            self.avg_peak_by_type_bytes[node_type] = (
                sum(composition_samples) / len(composition_samples)
                if composition_samples
                else 0.0
            )
            self.avg_max_by_type_bytes[node_type] = (
                sum(max_samples) / len(max_samples) if max_samples else 0.0
            )
        self.avg_torch_peak_bytes = (
            sum(self.torch_peak_bytes_per_run) / len(self.torch_peak_bytes_per_run)
            if self.torch_peak_bytes_per_run
            else 0.0
        )

    def print_stats(self) -> None:
        print("\n=== Graph Profiler Summary ===")
        print(f"Total graph nodes: {len(self.nodes)}")
        print(
            "Forward/Backward boundary: "
            f"forward_end_idx={self.forward_end_idx}, "
            f"backward_start_idx={self.backward_start_idx}, "
            f"optimizer_start_idx={self.optimizer_start_idx}"
        )
        print(f"Parameters detected: {len(self.param_placeholders)} placeholders")
        print(f"Optimizer states detected: {len(self.opt_state_placeholders)} placeholders")
        print(f"Gradients detected: {len(self.grad_placeholders)} placeholders")
        print(f"Activations detected: {len(self.activation_nodes)}")

        if self.activation_nodes:
            print("\nActivation lifetime analysis (first 20 by size):")
            activation_rank = sorted(
                self.activation_nodes,
                key=lambda n: self.avg_output_bytes.get(n, 0),
                reverse=True,
            )
            for node in activation_rank[:20]:
                last_fwd = self.last_forward_use[node]
                first_bwd = self.first_backward_use[node]
                print(
                    "  "
                    f"{node.name:<36} size={self._bytes_to_mib(self.avg_output_bytes.get(node, 0)):.3f} MiB "
                    f"last_fwd={last_fwd.name:<24} first_bwd={first_bwd.name}"
                )

        print("\nTop operators by average runtime (first 30):")
        ranked_nodes = sorted(
            self.nodes,
            key=lambda n: self.avg_runtime_ms.get(n, 0.0),
            reverse=True,
        )
        for node in ranked_nodes[:30]:
            node_type = self.node_types.get(node, NodeType.OTHER).name
            print(
                "  "
                f"{node.name:<36} op={node.op:<14} type={node_type:<6} "
                f"time={self.avg_runtime_ms.get(node, 0.0):8.4f} ms "
                f"out={self._bytes_to_mib(self.avg_output_bytes.get(node, 0)):8.3f} MiB "
                f"delta={self._bytes_to_mib(self.avg_memory_delta_bytes.get(node, 0.0)):8.3f} MiB"
            )

        print("\nPeak memory breakdown at total live-memory peak (live tensor model):")
        print(f"  total: {self._bytes_to_mib(self.avg_peak_total_bytes):.3f} MiB")
        for node_type in NodeType:
            print(
                "  "
                f"{node_type.name:<8}: "
                f"{self._bytes_to_mib(self.avg_peak_by_type_bytes[node_type]):.3f} MiB"
            )
        print(
            "\nPer-category maxima over iteration "
            "(sum can exceed total because peaks happen at different times):"
        )
        for node_type in NodeType:
            print(
                "  "
                f"{node_type.name:<8}: "
                f"{self._bytes_to_mib(self.avg_max_by_type_bytes[node_type]):.3f} MiB"
            )

        if torch.cuda.is_available():
            print(
                "Peak memory observed by torch.cuda.max_memory_allocated: "
                f"{self._bytes_to_mib(self.avg_torch_peak_bytes):.3f} MiB"
            )

    def reset_stats(self) -> None:
        # The statistics must be cleared out after x warm-up iterations and
        # reset before the actual measurement begins.
        self.node_runtime_ms: Dict[fx.Node, List[float]] = {
            node: [] for node in self.nodes
        }
        self.node_memory_delta_bytes: Dict[fx.Node, List[int]] = {
            node: [] for node in self.nodes
        }
        self.node_output_bytes: Dict[fx.Node, int] = {node: 0 for node in self.nodes}
        self.node_new_storage_bytes: Dict[fx.Node, int] = {
            node: 0 for node in self.nodes
        }

        self.peak_total_bytes_per_run: List[int] = []
        self.peak_total_composition_by_type_bytes_per_run: Dict[NodeType, List[int]] = {
            node_type: [] for node_type in NodeType
        }
        self.peak_by_type_bytes_per_run: Dict[NodeType, List[int]] = {
            node_type: [] for node_type in NodeType
        }
        self.torch_peak_bytes_per_run: List[int] = []
