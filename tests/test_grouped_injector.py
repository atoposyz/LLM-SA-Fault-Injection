"""
Unit tests for grouped-exact injector.

Verifies:
1. Additive-independence premise (fault-order permutation)
2. L0 exact vs L1 grouped_exact equivalence for all 6 slow paths
3. Multi-fault, multi-bit, overlapping scenarios
"""
import random
import sys

import torch

sys.path.insert(0, "tool/src")
from tool.ber_injector import BER_Fast_SA_FaultInjector
from tool.grouped_injector import GroupedExact_SA_FaultInjector

torch.manual_seed(42)
random.seed(42)

R, C = 256, 256
DTYPE = torch.float32
DEVICE = "cpu"


def make_injectors(dataflow, mode, precision="fp32"):
    """Create L0 (exact) and L1 (grouped_exact) injectors with same config."""
    ft = f"{mode}_stuck_1_0"
    l0 = BER_Fast_SA_FaultInjector(
        sa_rows=R, sa_cols=C, dataflow=dataflow,
        fault_type=ft, precision=precision)
    l0.enabled = True
    l1 = GroupedExact_SA_FaultInjector(
        sa_rows=R, sa_cols=C, dataflow=dataflow,
        fault_type=ft, precision=precision)
    l1.enabled = True
    return l0, l1


def set_faults(inj, rows, cols, bits, reg=0):
    """Manually set multi-fault positions on an injector."""
    inj.reset_fault_pe()
    for r, c, b in zip(rows, cols, bits):
        inj.fault_pe_row.append(r)
        inj.fault_pe_col.append(c)
        inj.fault_bit.append(b)
        inj.fault_reg.append(reg)
    inj._fault_generation += 1  # invalidate cache


def run_hook(inj, X, W):
    """Simulate one hook call. Returns output tensor.
    X: [M, K], W: [N, K] (weight in linear layer format [out, in])."""
    class MockModule:
        pass
    m = MockModule()
    m.weight = W.contiguous()  # [N, K]; hook_fn does .T to get [K, N]
    out = X @ W.T
    result = inj.hook_fn(m, (X,), (out,))
    if isinstance(result, tuple):
        return result[0]
    return result


def test_fault_order_invariance():
    """Step 1: Verify additive-independence — permuting fault order
    produces identical results within FP rounding."""
    print("=== Fault-order permutation test ===")
    dataflows_modes = [
        ("OS", "input"), ("OS", "weight"),
        ("WS", "input"), ("WS", "psum"),
        ("IS", "weight"), ("IS", "psum"),
    ]
    M, K, N = 4, 128, 64
    X = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    W = torch.randn(N, K, dtype=DTYPE, device=DEVICE)

    for df, mode in dataflows_modes:
        l0, _ = make_injectors(df, mode)

        # Generate 8 random faults
        rows = [random.randint(0, R - 1) for _ in range(8)]
        cols = [random.randint(0, C - 1) for _ in range(8)]
        bits = [random.randint(0, 15) for _ in range(8)]

        # Forward order
        set_faults(l0, rows, cols, bits)
        out_fwd = run_hook(l0, X, W)

        # Reversed order
        set_faults(l0, list(reversed(rows)), list(reversed(cols)),
                   list(reversed(bits)))
        out_rev = run_hook(l0, X, W)

        # Shuffled order
        idx = list(range(8))
        random.shuffle(idx)
        set_faults(l0, [rows[i] for i in idx], [cols[i] for i in idx],
                   [bits[i] for i in idx])
        out_shuf = run_hook(l0, X, W)

        torch.testing.assert_close(out_fwd, out_rev,
                                   msg=f"{df}+{mode}: fwd vs rev mismatch")
        torch.testing.assert_close(out_fwd, out_shuf,
                                   msg=f"{df}+{mode}: fwd vs shuffle mismatch")
        print(f"  {df}+{mode}: OK")


def test_single_fault():
    """Single fault: L0 == L1."""
    print("\n=== Single fault tests ===")
    dataflows_modes = [
        ("OS", "input"), ("OS", "weight"), ("OS", "psum"),
        ("WS", "input"), ("WS", "weight"), ("WS", "psum"),
        ("IS", "input"), ("IS", "weight"), ("IS", "psum"),
    ]
    M, K, N = 3, 127, 59  # non-multiples of 256
    X = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    W = torch.randn(N, K, dtype=DTYPE, device=DEVICE)

    for df, mode in dataflows_modes:
        l0, l1 = make_injectors(df, mode)
        r, c, b = random.randint(10, 100), random.randint(10, 100), 5
        set_faults(l0, [r], [c], [b])
        set_faults(l1, [r], [c], [b])

        out0 = run_hook(l0, X, W)
        out1 = run_hook(l1, X, W)

        torch.testing.assert_close(out0, out1,
                                   msg=f"{df}+{mode}: single fault mismatch")
        print(f"  {df}+{mode}: OK")


def _make_overlapping_faults():
    """Generate overlapping faults: same PE, same bit."""
    rows = [50, 50, 51, 51]
    cols = [30, 30, 31, 31]
    bits = [7, 7, 8, 8]
    return rows, cols, bits


def test_overlapping_same_mask():
    """Overlapping faults with same mask → count > 1."""
    print("\n=== Overlapping same-mask tests ===")
    dataflows_modes = [
        ("OS", "input"), ("OS", "weight"),
        ("WS", "input"), ("WS", "psum"),
        ("IS", "weight"), ("IS", "psum"),
    ]
    M, K, N = 4, 200, 150
    X = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    W = torch.randn(N, K, dtype=DTYPE, device=DEVICE)
    rows, cols, bits = _make_overlapping_faults()

    for df, mode in dataflows_modes:
        l0, l1 = make_injectors(df, mode)
        set_faults(l0, rows, cols, bits)
        set_faults(l1, rows, cols, bits)

        out0 = run_hook(l0, X, W)
        out1 = run_hook(l1, X, W)

        torch.testing.assert_close(out0, out1,
                                   msg=f"{df}+{mode}: overlapping same mask")
        print(f"  {df}+{mode}: OK")


def test_different_masks():
    """Faults with different bitmasks in overlapping regions."""
    print("\n=== Different-mask tests ===")
    dataflows_modes = [
        ("OS", "input"), ("OS", "weight"),
        ("WS", "input"), ("WS", "psum"),
        ("IS", "weight"), ("IS", "psum"),
    ]
    M, K, N = 4, 200, 150
    X = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    W = torch.randn(N, K, dtype=DTYPE, device=DEVICE)

    rows = [50, 50, 51]
    cols = [30, 30, 31]
    bits = [5, 10, 15]  # different bits

    for df, mode in dataflows_modes:
        l0, l1 = make_injectors(df, mode)
        set_faults(l0, rows, cols, bits)
        set_faults(l1, rows, cols, bits)

        out0 = run_hook(l0, X, W)
        out1 = run_hook(l1, X, W)

        torch.testing.assert_close(out0, out1,
                                   msg=f"{df}+{mode}: different masks")
        print(f"  {df}+{mode}: OK")


def test_multi_bit_same_pe():
    """Same PE, multiple fault bits → unique mask is multi-bit."""
    print("\n=== Multi-bit same PE tests ===")
    dataflows_modes = [
        ("OS", "input"), ("OS", "weight"),
        ("WS", "input"), ("WS", "psum"),
        ("IS", "weight"), ("IS", "psum"),
    ]
    M, K, N = 4, 200, 150
    X = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    W = torch.randn(N, K, dtype=DTYPE, device=DEVICE)

    rows = [60, 60]
    cols = [40, 40]
    bits = [3, 7]  # same PE, different bits → merged mask

    for df, mode in dataflows_modes:
        l0, l1 = make_injectors(df, mode)
        set_faults(l0, rows, cols, bits)
        set_faults(l1, rows, cols, bits)

        out0 = run_hook(l0, X, W)
        out1 = run_hook(l1, X, W)

        torch.testing.assert_close(out0, out1,
                                   msg=f"{df}+{mode}: multi-bit same PE")
        print(f"  {df}+{mode}: OK")


def test_shape_variants():
    """Test prefill (M>1) and decode (M=1) shapes, plus K/N > and < SA dims."""
    print("\n=== Shape variant tests ===")
    shapes = [
        # (M, K, N) — prefill shapes
        (8, 512, 256),                          # K,N near SA dims
        (32, 4096, 4096),                       # large, typical LLM
        (1, 128, 64),                           # decode, small
        (1, 200, 300),                          # decode, K<SA, N>SA
        (5, 100, 50),                           # all < SA dims
    ]
    mode_pairs = [
        ("OS", "input"), ("OS", "weight"),
        ("WS", "input"), ("WS", "psum"),
        ("IS", "weight"), ("IS", "psum"),
    ]

    for M, K, N in shapes:
        X = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
        W = torch.randn(N, K, dtype=DTYPE, device=DEVICE)
        for df, mode in mode_pairs:
            l0, l1 = make_injectors(df, mode)
            rows = [random.randint(0, R - 1) for _ in range(10)]
            cols = [random.randint(0, C - 1) for _ in range(10)]
            bits = [random.randint(0, 15) for _ in range(10)]
            set_faults(l0, rows, cols, bits)
            set_faults(l1, rows, cols, bits)

            out0 = run_hook(l0, X, W)
            out1 = run_hook(l1, X, W)

            torch.testing.assert_close(
                out0, out1, rtol=1e-5, atol=1e-7,
                msg=f"{df}+{mode} shape {M}x{K}x{N}")
        print(f"  shape {M}x{K}x{N}: OK ({len(mode_pairs)} modes)")


def test_high_ber_simulation():
    """Simulate BER=1e-4 with 200 random faults."""
    print("\n=== High-BER simulation ===")
    dataflows_modes = [
        ("OS", "input"), ("OS", "weight"),
        ("WS", "input"), ("WS", "psum"),
        ("IS", "weight"), ("IS", "psum"),
    ]
    M, K, N = 4, 2048, 2048
    X = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    W = torch.randn(N, K, dtype=DTYPE, device=DEVICE)

    num_faults = 200
    rows = [random.randint(0, R - 1) for _ in range(num_faults)]
    cols = [random.randint(0, C - 1) for _ in range(num_faults)]
    bits = [random.randint(0, 15) for _ in range(num_faults)]

    for df, mode in dataflows_modes:
        l0, l1 = make_injectors(df, mode)
        set_faults(l0, rows, cols, bits)
        set_faults(l1, rows, cols, bits)

        out0 = run_hook(l0, X, W)
        out1 = run_hook(l1, X, W)

        torch.testing.assert_close(out0, out1, rtol=1e-2, atol=1e-1,
                                   msg=f"{df}+{mode}: high BER")
        print(f"  {df}+{mode}: OK ({num_faults} faults)")


if __name__ == "__main__":
    test_fault_order_invariance()
    test_single_fault()
    test_overlapping_same_mask()
    test_different_masks()
    test_multi_bit_same_pe()
    test_shape_variants()
    test_high_ber_simulation()
    print("\n=== All tests passed ===")
