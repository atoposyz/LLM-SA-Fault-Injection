"""
Propagation-limited SA fault injector.

Extends Fast_SA_FaultInjector with a propagation_degree parameter that limits
how many systolic array columns an input-mode fault propagates to in the WS
dataflow. By default (propagation_degree=256), behavior matches the parent class
unconditionally propagating to all columns to the right of the fault column.
"""
import torch

from tool.fault_injector_next import Fast_SA_FaultInjector


class Propagation_SA_FaultInjector(Fast_SA_FaultInjector):
    """
    SA fault injector with configurable WS input-mode propagation distance.

    When fault_config['mode'] == 'input', only columns within
    [fault_column, fault_column + propagation_degree] are affected.
    Weight and psum modes delegate to the parent implementation.
    """
    def __init__(self, propagation_degree=256, **kwargs):
        super().__init__(**kwargs)
        self.propagation_degree = propagation_degree

    def _simulate_ws(self, X, W, M, K, N, r_f, c_f, device):
        if self.fault_config['mode'] != 'input':
            return super()._simulate_ws(X, W, M, K, N, r_f, c_f, device)

        # ---- input-mode (replicated from parent with propagation limit) ----
        k_mod = (torch.arange(K, device=device) % self.sa_rows).unsqueeze(1)  # [K, 1]
        j_mod = (torch.arange(N, device=device) % self.sa_cols).unsqueeze(0)  # [1, N]

        mode = self.fault_config['mode']
        op = self.fault_config['op']
        pe_mask_map = self._build_pe_mask_map(device)  # [R, C]

        Y_tilde = torch.matmul(X, W)
        # Find columns in the PE array that have faults
        active_cols = torch.nonzero(pe_mask_map.sum(dim=0)).squeeze(-1)
        if active_cols.dim() == 0:
            active_cols = active_cols.unsqueeze(0)

        for c in active_cols:
            col_mask = pe_mask_map[:, c]  # [R]
            k_mask_for_c = col_mask[k_mod.squeeze(-1)]  # [K]

            # Active bits on X for this column's propagation
            active_k = k_mask_for_c > 0 if op != 'stuck_0' else k_mask_for_c != 0
            if not active_k.any():
                continue

            # We apply the error uniquely to the elements needing it
            X_flipped = X.clone()
            # Expand mask for batch M assuming broadcast [M, K]
            mask_2d = k_mask_for_c.unsqueeze(0).expand(M, K)

            # Only operate on active features to save memory bandwidth if sparse
            active_2d = active_k.unsqueeze(0).expand(M, K)
            X_flipped[active_2d] = self._inject_bit_error(
                X_flipped[active_2d], mask_2d[active_2d], op
            )

            dx = X_flipped - X  # [M, K]

            in_mask = (j_mod >= c) & (j_mod < c + self.propagation_degree + 1)
            in_mask = in_mask.squeeze(0)  # [N]

            W_masked = W * in_mask.to(W.dtype)  # [K, N]
            Y_tilde += torch.matmul(dx, W_masked)

        return Y_tilde
