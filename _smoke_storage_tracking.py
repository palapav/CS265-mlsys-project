"""Focused unit smoke tests for the storage-based liveness model.

We can't trace a foreach-Adam train step on CPU (capturable=True needs CUDA;
capturable=False calls .item() on the step counter inside FakeTensorMode and
fails). So instead we directly exercise the two parts of GraphProfiler that
broke before:

  1. `_extract_storages` collapses aliases by underlying storage.
  2. The `run_node` refcount/live model never exceeds the true unique-storage
     bytes when fed alias-heavy inputs.

The second test bypasses fx tracing and drives GraphProfiler with a tiny
hand-rolled graph whose tensors deliberately share storage (mimicking what
foreach + getitem + view do in a real Adam step).
"""
from typing import List

import torch
import torch.fx as fx

from graph_prof import GraphProfiler, NodeType


def _bytes(t: torch.Tensor) -> int:
    return int(t.untyped_storage().nbytes())


def test_extract_storages_dedup():
    base = torch.empty(1024, dtype=torch.float32)
    view_a = base.view(32, 32)
    view_b = base[:512]
    transposed = view_a.t()

    nested = {"x": [base, view_a], "y": (view_b,), "z": transposed}

    prof = _BareProfiler()
    storages = prof._extract_storages(nested)

    assert len(storages) == 1, f"expected 1 unique storage, got {len(storages)}: {storages}"
    only = next(iter(storages.values()))
    assert only == _bytes(base), f"expected {_bytes(base)}, got {only}"


def test_extract_storages_distinct():
    a = torch.empty(8, dtype=torch.float32)
    b = torch.empty(16, dtype=torch.float32)
    c = a.view(4, 2)

    prof = _BareProfiler()
    storages = prof._extract_storages([a, b, c])
    assert len(storages) == 2
    sizes = sorted(storages.values())
    assert sizes == sorted([_bytes(a), _bytes(b)])


def test_storage_tracking_aliases_not_double_counted():
    """Hand-built fx graph that mimics:

        opt_state = placeholder()  # 4 KiB
        param = placeholder()      # 4 KiB
        # foreach-style: a list of two tensors aliasing the inputs (in real
        # Adam, the foreach op produces fresh tensors, but for stress-testing
        # the alias dedup we make these aliases of the placeholders).
        items = [opt_state.view_as(opt_state), param.view_as(param)]
        a = items[0]               # alias of opt_state
        b = items[1]               # alias of param
        new_state = a.clone()      # NEW storage, 4 KiB
        opt_state.copy_(new_state) # in-place, no new storage
        param.copy_(b)             # in-place, no new storage
        return None

    Naive per-node accounting double-counts the placeholder storages via the
    aliasing nodes. Storage-based accounting must report total <= 3 * 4 KiB
    (param + opt_state + new_state) at any point.
    """

    nbytes = 1024 * 4  # 1024 floats per tensor

    class _M(torch.nn.Module):
        def forward(self, opt_state, param):
            items = [opt_state.view_as(opt_state), param.view_as(param)]
            a = items[0]
            b = items[1]
            new_state = a.clone()
            opt_state.copy_(new_state)
            param.copy_(b)
            return new_state

    gm = fx.symbolic_trace(_M())
    profiler = GraphProfiler(gm)
    opt_state = torch.empty(1024, dtype=torch.float32)
    param = torch.empty(1024, dtype=torch.float32)

    profiler.reset_stats()
    profiler.run(opt_state, param)
    profiler.aggregate_stats()

    peak = profiler.avg_peak_total_bytes
    composition = {nt.name: profiler.avg_peak_by_type_bytes[nt] for nt in NodeType}
    composition_sum = sum(composition.values())

    print(f"  peak_live bytes        : {peak}")
    print(f"  peak_live MiB          : {peak / (1024*1024):.6f}")
    print(f"  composition (bytes)    : {composition}")
    print(f"  composition sum bytes  : {composition_sum}")

    # 3 unique storages exist simultaneously: opt_state, param, new_state.
    expected_max_bytes = 3 * nbytes
    assert peak <= expected_max_bytes, (
        f"peak_live {peak} exceeds physical upper bound {expected_max_bytes}; "
        f"alias dedup is not working"
    )
    assert peak >= 2 * nbytes, (
        f"peak_live {peak} is unreasonably small; expected >= {2*nbytes}"
    )
    assert abs(composition_sum - peak) <= 1, "breakdown does not sum to total"


class _BareProfiler:
    """Minimal stand-in just to call `_extract_storages` without instantiating
    a full fx graph; we copy the method bound to a fake `self`."""

    _extract_storages = GraphProfiler._extract_storages


def main():
    print("test_extract_storages_dedup ...")
    test_extract_storages_dedup()
    print("  ok")
    print("test_extract_storages_distinct ...")
    test_extract_storages_distinct()
    print("  ok")
    print("test_storage_tracking_aliases_not_double_counted ...")
    test_storage_tracking_aliases_not_double_counted()
    print("  ok")
    print("\nAll storage-tracking smoke tests passed.")


if __name__ == "__main__":
    main()
