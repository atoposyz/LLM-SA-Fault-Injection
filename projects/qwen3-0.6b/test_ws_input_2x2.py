from pathlib import Path
import sys

import torch


FAULTINJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FAULTINJECT_ROOT / "tool" / "src"))

from tool.fault_injector_next import Fast_SA_FaultInjector


def main():
    torch.set_printoptions(precision=6, sci_mode=False)

    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    W = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)

    injector = Fast_SA_FaultInjector(
        sa_rows=2,
        sa_cols=2,
        dataflow="WS",
        fault_type="input_bitflip_23",
        precision="fp32",
    )
    injector.set_fault_position(row=0, col=0)

    M, K = X.shape
    _, N = W.shape
    device = X.device
    dummy_r = torch.tensor(injector.fault_pe_row, device=device).view(-1, 1, 1)
    dummy_c = torch.tensor(injector.fault_pe_col, device=device).view(-1, 1, 1)

    Y_clean = torch.matmul(X, W)
    Y_faulty = injector._simulate_ws(X, W, M, K, N, dummy_r, dummy_c, device)

    # WS input fault at PE(row=0, col=0) affects k where k % 2 == 0.
    # For float32, flipping bit 23 changes X[:, 0] from [1, 3] to [0.5, 6].
    X_expected_faulty = torch.tensor([[0.5, 2.0], [6.0, 4.0]], dtype=torch.float32)
    Y_expected = torch.matmul(X_expected_faulty, W)

    print("X:")
    print(X)
    print("\nW:")
    print(W)
    print("\nClean Y = X @ W:")
    print(Y_clean)
    print("\nFault config: WS input_bitflip_23 at PE(row=0, col=0)")
    print("Expected faulty X used for affected propagation:")
    print(X_expected_faulty)
    print("\nFaulty Y from injector:")
    print(Y_faulty)
    print("\nExpected faulty Y:")
    print(Y_expected)
    print("\nDelta Y:")
    print(Y_faulty - Y_clean)
    print("\nMatches expected:", torch.allclose(Y_faulty, Y_expected))


if __name__ == "__main__":
    main()
