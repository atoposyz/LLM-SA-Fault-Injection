"""
Grouped-exact SA fault injector.

Replaces per-PE-point loops with count-map grouped accumulation for stuck-at
faults.  Mathematically equivalent to the additive baseline (up to FP rounding),
with O(U) matmuls/injections instead of O(num_faults).

Bitflip faults are rejected — XOR semantics (double-flip cancels) are
incompatible with OR-aggregation.
"""

import hashlib
import struct

import torch

from tool.ber_injector import BER_Fast_SA_FaultInjector


def _pe_mask_digest(pe_mask_map, stuck_direction: str, dataflow: str,
                    mode: str, sa_rows: int, sa_cols: int,
                    precision: str) -> str:
    """Stable digest of fault configuration for cache keying."""
    h = hashlib.sha256()
    h.update(pe_mask_map.cpu().numpy().tobytes())
    h.update(stuck_direction.encode())
    h.update(dataflow.encode())
    h.update(mode.encode())
    h.update(struct.pack("ii", sa_rows, sa_cols))
    h.update(precision.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pre-computed count maps (once per trial)
# ---------------------------------------------------------------------------


class GroupedPrecomputed:
    """Count maps derived from pe_mask_map for a single (dataflow, mode) pair."""

    __slots__ = ("unique_masks", "dataflow", "mode",
                 # OS input
                 "P_in_m",
                 # OS weight / IS weight
                 "P_w_m",
                 # WS input
                 "col_vectors", "h_v", "v_masks",
                 # WS psum
                 "n_ws_m",
                 # IS psum
                 "n_is_m",
                 "digest")

    def __init__(self, pe_mask_map, dataflow: str, mode: str,
                 stuck_direction: str, sa_rows: int = 256, sa_cols: int = 256,
                 precision: str = "bf16"):
        self.dataflow = dataflow
        self.mode = mode

        nonzero_mask = pe_mask_map != 0
        unique_vals = torch.unique(pe_mask_map[nonzero_mask])
        self.unique_masks = [int(v.item()) for v in unique_vals]

        self.digest = _pe_mask_digest(
            pe_mask_map, stuck_direction, dataflow, mode, sa_rows, sa_cols,
            precision)

        if dataflow == "OS":
            if mode == "input":
                self._build_os_input(pe_mask_map, sa_rows, sa_cols)
            elif mode == "weight":
                self._build_os_weight(pe_mask_map, sa_rows, sa_cols)
        elif dataflow == "WS":
            if mode == "input":
                self._build_ws_input(pe_mask_map, sa_rows, sa_cols)
            elif mode == "psum":
                self._build_ws_psum(pe_mask_map, sa_rows, sa_cols)
        elif dataflow == "IS":
            if mode == "weight":
                self._build_is_weight(pe_mask_map, sa_rows, sa_cols)
            elif mode == "psum":
                self._build_is_psum(pe_mask_map, sa_rows, sa_cols)

    # ---- OS input --------------------------------------------------------

    def _build_os_input(self, pmap, R, C):
        self.P_in_m = {}
        for m in self.unique_masks:
            eq = (pmap == m).int()          # [R, C]
            self.P_in_m[m] = eq.cumsum(dim=1)  # row-wise prefix count

    # ---- OS weight / IS weight -------------------------------------------

    def _build_os_weight(self, pmap, R, C):
        self.P_w_m = {}
        for m in self.unique_masks:
            eq = (pmap == m).int()
            self.P_w_m[m] = eq.cumsum(dim=0)  # column-wise prefix count

    _build_is_weight = _build_os_weight  # same prefix structure

    # ---- WS input --------------------------------------------------------

    def _build_ws_input(self, pmap, R, C):
        self.col_vectors = {}   # bytes key → list of column indices
        self.v_masks = {}        # bytes key → k_mask [R] int32
        self.h_v = {}            # bytes key → [C] int32 count prefix

        pmap_cpu = pmap.cpu()
        for c in range(C):
            col = pmap_cpu[:, c]
            if col.eq(0).all():
                continue
            key = col.numpy().tobytes()
            self.col_vectors.setdefault(key, []).append(c)
            if key not in self.v_masks:
                self.v_masks[key] = col.clone().to(torch.int32)

        for key, cols in self.col_vectors.items():
            count = torch.zeros(C, dtype=torch.int32)
            for c in cols:
                count[c] = 1
            self.h_v[key] = count.cumsum(dim=0)

    # ---- WS psum ---------------------------------------------------------

    def _build_ws_psum(self, pmap, R, C):
        self.n_ws_m = {}
        for m in self.unique_masks:
            self.n_ws_m[m] = (pmap == m).int().sum(dim=0)  # [C] count per col

    # ---- IS psum ---------------------------------------------------------

    def _build_is_psum(self, pmap, R, C):
        self.n_is_m = {}
        for m in self.unique_masks:
            self.n_is_m[m] = (pmap == m).int().sum(dim=1)  # [R] count per row


# ---------------------------------------------------------------------------
# Shape-dependent expanded cache
# ---------------------------------------------------------------------------


class ExpandedCache:
    """Caches modulo-expanded tensors for a specific (M, N, K, device)."""

    __slots__ = ("i_mod", "j_mod", "k_mod", "M", "N", "K", "device")

    def __init__(self, M, N, K, device, sa_rows=256, sa_cols=256):
        self.M = M
        self.N = N
        self.K = K
        self.device = device
        self.i_mod = torch.arange(M, device=device) % sa_rows   # [M]
        self.j_mod = torch.arange(N, device=device) % sa_cols   # [N]
        self.k_mod = torch.arange(K, device=device) % sa_rows   # [K]


# ---------------------------------------------------------------------------
# Grouped-exact injector
# ---------------------------------------------------------------------------


class GroupedExact_SA_FaultInjector(BER_Fast_SA_FaultInjector):
    """L1 grouped-exact injector for stuck-at faults.

    Overrides slow per-PE simulation paths with count-map accumulation.
    Fast paths (WS weight, OS psum, IS input) delegate to parent.
    Bitflip faults fall back to parent L0 exact.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grouped_cache = {}   # (dataflow, mode) → GroupedPrecomputed
        self._expanded_cache = {}  # (digest, M, N, K, device) → ExpandedCache
        self._use_grouped = True   # set False to force L0 fallback

    # ---- cache management ------------------------------------------------

    def reset_fault_pe(self):
        super().reset_fault_pe()
        self._grouped_cache.clear()
        self._expanded_cache.clear()

    def _ensure_grouped(self, device, M, N, K):
        dataflow = self.dataflow
        mode = self.fault_config["mode"]
        op = self.fault_config["op"]

        if self._use_grouped and "flip" in op:
            self._use_grouped = False

        if not self._use_grouped:
            return None, None

        pe_mask_map = self._build_pe_mask_map(device)

        key = (dataflow, mode)
        if key not in self._grouped_cache:
            self._grouped_cache[key] = GroupedPrecomputed(
                pe_mask_map, dataflow, mode, op,
                sa_rows=self.sa_rows, sa_cols=self.sa_cols,
                precision=self.precision)

        g = self._grouped_cache[key]
        ec_key = (g.digest, M, N, K, str(device))
        if ec_key not in self._expanded_cache:
            self._expanded_cache[ec_key] = ExpandedCache(
                M, N, K, device, self.sa_rows, self.sa_cols)

        return g, self._expanded_cache[ec_key]

    # ---- hook ------------------------------------------------------------

    def hook_fn(self, module, input_tuple, output_tuple):
        if not self.enabled or not self.fault_pe_row:
            return output_tuple

        X = input_tuple[0].contiguous()
        W = module.weight.T.contiguous()
        original_shape = X.shape
        X_2d = X.view(-1, original_shape[-1])
        device = W.device
        M, K = X_2d.shape
        _, N = W.shape

        g, ec = self._ensure_grouped(device, M, N, K)

        if self.dataflow == "WS":
            Y_faulty = self._simulate_ws_grouped(X_2d, W, M, K, N, device, g, ec)
        elif self.dataflow == "OS":
            Y_faulty = self._simulate_os_grouped(X_2d, W, M, K, N, device, g, ec)
        elif self.dataflow == "IS":
            Y_faulty = self._simulate_is_grouped(X_2d, W, M, K, N, device, g, ec)
        else:
            raise ValueError(f"Unknown dataflow: {self.dataflow}")

        Y_faulty = torch.clamp(Y_faulty, -1e3, 1e3)
        return Y_faulty.view(original_shape[:-1] + (N,))

    # ---- WS simulation ---------------------------------------------------

    def _simulate_ws_grouped(self, X, W, M, K, N, device, g, ec):
        mode = self.fault_config["mode"]
        op = self.fault_config["op"]

        if mode == "weight":
            return self._simulate_ws(X, W, M, K, N, device)  # parent (fast)

        if g is None:
            return self._simulate_ws(X, W, M, K, N, device)  # L0 fallback

        if mode == "input":
            return self._ws_input_grouped(X, W, M, K, N, device, g, ec, op)
        elif mode == "psum":
            return self._ws_psum_grouped(X, W, M, K, N, device, g, ec, op)
        return self._simulate_ws(X, W, M, K, N, device)

    def _ws_input_grouped(self, X, W, M, K, N, device, g, ec, op):
        Y = torch.matmul(X, W)
        for key, cols in g.col_vectors.items():
            vm = g.v_masks[key].to(device)          # [R] int32
            hv = g.h_v[key].to(device)              # [C] int32

            k_mask = vm[ec.k_mod]                   # [K] int32
            active = k_mask != 0
            if not active.any():
                continue

            X_corr = X.clone()
            mask_2d = k_mask.unsqueeze(0).expand(M, K)
            active_2d = active.unsqueeze(0).expand(M, K)
            X_corr[active_2d] = self._inject_bit_error(
                X_corr[active_2d], mask_2d[active_2d], op)
            dX = X_corr - X

            w_count = hv[ec.j_mod].unsqueeze(0).to(W.dtype)   # [1, N]
            Y += torch.matmul(dX, W * w_count)
        return Y

    def _ws_psum_grouped(self, X, W, M, K, N, device, g, ec, op):
        Y = torch.matmul(X, W)
        for m in g.unique_masks:
            n = g.n_ws_m[m].to(device)              # [C]
            count_2d = n[ec.j_mod].unsqueeze(0)      # [1, N]
            active_cols = count_2d > 0
            if not active_cols.any():
                continue

            Y_corr = Y.clone()
            mask_t = Y_corr.new_full((1,), m, dtype=torch.int32).expand(M, N)
            Y_corr[active_cols.expand(M, N)] = self._inject_bit_error(
                Y_corr[active_cols.expand(M, N)],
                mask_t[active_cols.expand(M, N)], op)
            dY = Y_corr - Y
            Y += dY * count_2d.to(Y.dtype)
        return Y

    # ---- OS simulation ---------------------------------------------------

    def _simulate_os_grouped(self, X, W, M, K, N, device, g, ec):
        mode = self.fault_config["mode"]
        op = self.fault_config["op"]

        if mode == "psum":
            return self._simulate_os(X, W, M, K, N, device)  # parent (fast)

        if g is None:
            return self._simulate_os(X, W, M, K, N, device)  # L0 fallback

        if mode == "input":
            return self._os_input_grouped(X, W, M, K, N, device, g, ec, op)
        elif mode == "weight":
            return self._os_weight_grouped(X, W, M, K, N, device, g, ec, op)
        return self._simulate_os(X, W, M, K, N, device)

    def _os_input_grouped(self, X, W, M, K, N, device, g, ec, op):
        Y = torch.matmul(X, W)
        for m in g.unique_masks:
            P = g.P_in_m[m].to(device)              # [R, C]
            count_2d = P[ec.i_mod.unsqueeze(1),
                         ec.j_mod.unsqueeze(0)]      # [M, N]
            active = count_2d > 0
            if not active.any():
                continue

            X_corr = X.clone()
            mask_t = X_corr.new_full((1,), m, dtype=torch.int32).expand(M, K)
            X_corr = self._inject_bit_error(X_corr, mask_t, op)
            dY = torch.matmul(X_corr, W) - Y
            Y += dY * count_2d.to(Y.dtype)
        return Y

    def _os_weight_grouped(self, X, W, M, K, N, device, g, ec, op):
        Y = torch.matmul(X, W)
        for m in g.unique_masks:
            P = g.P_w_m[m].to(device)              # [R, C]
            count_2d = P[ec.i_mod.unsqueeze(1),
                         ec.j_mod.unsqueeze(0)]      # [M, N]
            active = count_2d > 0
            if not active.any():
                continue

            W_corr = W.clone()
            mask_t = W_corr.new_full((1,), m, dtype=torch.int32).expand(K, N)
            W_corr = self._inject_bit_error(W_corr, mask_t, op)
            dY = torch.matmul(X, W_corr) - Y
            Y += dY * count_2d.to(Y.dtype)
        return Y

    # ---- IS simulation ---------------------------------------------------

    def _simulate_is_grouped(self, X, W, M, K, N, device, g, ec):
        mode = self.fault_config["mode"]
        op = self.fault_config["op"]

        if mode == "input":
            return self._simulate_is(X, W, M, K, N, device)  # parent (fast)

        if g is None:
            return self._simulate_is(X, W, M, K, N, device)  # L0 fallback

        if mode == "weight":
            return self._is_weight_grouped(X, W, M, K, N, device, g, ec, op)
        elif mode == "psum":
            return self._is_psum_grouped(X, W, M, K, N, device, g, ec, op)
        return self._simulate_is(X, W, M, K, N, device)

    def _is_weight_grouped(self, X, W, M, K, N, device, g, ec, op):
        Y = torch.matmul(X, W)
        for m in g.unique_masks:
            P = g.P_w_m[m].to(device)              # [R, C]
            T_2d = P[ec.i_mod.unsqueeze(1),
                     ec.k_mod.unsqueeze(0)]          # [M, K]
            active = T_2d > 0
            if not active.any():
                continue

            W_corr = W.clone()
            mask_t = W_corr.new_full((1,), m, dtype=torch.int32).expand(K, N)
            W_corr = self._inject_bit_error(W_corr, mask_t, op)
            dW = W_corr - W

            X_weighted = X * T_2d.to(X.dtype)
            Y += torch.matmul(X_weighted, dW)
        return Y

    def _is_psum_grouped(self, X, W, M, K, N, device, g, ec, op):
        Y = torch.matmul(X, W)
        for m in g.unique_masks:
            n = g.n_is_m[m].to(device)              # [R]
            count_2d = n[ec.i_mod].unsqueeze(1)      # [M, 1]
            active_rows = (count_2d > 0).squeeze(-1)
            if not active_rows.any():
                continue

            Y_corr = Y.clone()
            mask_t = Y_corr.new_full((1,), m, dtype=torch.int32).expand(M, N)
            Y_corr[active_rows.unsqueeze(1).expand(M, N)] = self._inject_bit_error(
                Y_corr[active_rows.unsqueeze(1).expand(M, N)],
                mask_t[active_rows.unsqueeze(1).expand(M, N)], op)
            dY = Y_corr - Y
            Y += dY * count_2d.to(Y.dtype)
        return Y


# ---------------------------------------------------------------------------
# Global-coverage injector — O(1) per hook independent of fault count
# ---------------------------------------------------------------------------


class _GlobalCoverage:
    """Pre-computed global-coverage masks. Built once per trial."""

    __slots__ = ("global_mask", "S_template", "dataflow", "mode",
                 "row_mask", "col_mask", "c_min", "r_min", "digest")

    def __init__(self, pe_mask_map, dataflow, mode, stuck_direction,
                 sa_rows=256, sa_cols=256, precision="bf16"):
        self.dataflow = dataflow
        self.mode = mode

        # Global bitmask: OR of all PE masks
        g = 0
        for val in pe_mask_map.flatten().tolist():
            g |= val
        self.global_mask = g

        if g == 0:
            return

        S = torch.zeros(sa_rows, sa_cols, dtype=torch.bool)

        if dataflow == "OS":
            if mode == "input":
                for r in range(sa_rows):
                    col = pe_mask_map[r, :]
                    nz = (col != 0).nonzero(as_tuple=False)
                    if len(nz) > 0:
                        c_min = nz[0].item()
                        S[r, c_min:] = True
                self.S_template = S

            elif mode == "weight":
                for c in range(sa_cols):
                    col = pe_mask_map[:, c]
                    nz = (col != 0).nonzero(as_tuple=False)
                    if len(nz) > 0:
                        r_min = nz[0].item()
                        S[r_min:, c] = True
                self.S_template = S

        elif dataflow == "WS":
            if mode == "input":
                self.row_mask = torch.zeros(sa_rows, dtype=torch.int32)
                self.c_min = torch.full((sa_rows,), sa_cols, dtype=torch.int32)
                for r in range(sa_rows):
                    col = pe_mask_map[r, :]
                    nz = (col != 0).nonzero(as_tuple=False)
                    if len(nz) > 0:
                        self.row_mask[r] = g
                        self.c_min[r] = nz[0].item()

            elif mode == "psum":
                self.col_mask = (pe_mask_map.sum(dim=0) != 0)  # [C] bool

        elif dataflow == "IS":
            if mode == "weight":
                self.r_min = torch.full((sa_cols,), sa_rows, dtype=torch.int32)
                self.col_mask = torch.zeros(sa_cols, dtype=torch.bool)
                for c in range(sa_cols):
                    col = pe_mask_map[:, c]
                    nz = (col != 0).nonzero(as_tuple=False)
                    if len(nz) > 0:
                        self.col_mask[c] = True
                        self.r_min[c] = nz[0].item()
                for c in range(sa_cols):
                    if self.col_mask[c]:
                        S[self.r_min[c]:, c] = True
                self.S_template = S

            elif mode == "psum":
                self.row_mask = (pe_mask_map.sum(dim=1) != 0)  # [R] bool

        self.digest = _pe_mask_digest(
            pe_mask_map, stuck_direction, dataflow, mode, sa_rows, sa_cols,
            precision)


class GlobalCoverage_SA_FaultInjector(BER_Fast_SA_FaultInjector):
    """Global-coverage injector: 1 injection + ≤1 matmul per hook.

    All fault bitmasks are OR'd into a single global mask; spatial regions
    are unioned.  Independent of fault count — same cost at BER=1e-4 as at
    single-fault.  A spatial-and-bitmask over-approximation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cov_cache = {}     # (dataflow, mode) → _GlobalCoverage
        self._expanded_cache = {}  # (digest, M, N, K, device) → ExpandedCache

    def reset_fault_pe(self):
        super().reset_fault_pe()
        self._cov_cache.clear()
        self._expanded_cache.clear()

    def _ensure_coverage(self, device, M, N, K):
        dataflow = self.dataflow
        mode = self.fault_config["mode"]
        op = self.fault_config["op"]

        if "flip" in op:
            return None, None

        pe_mask_map = self._build_pe_mask_map(device)

        key = (dataflow, mode)
        if key not in self._cov_cache:
            self._cov_cache[key] = _GlobalCoverage(
                pe_mask_map, dataflow, mode, op,
                sa_rows=self.sa_rows, sa_cols=self.sa_cols,
                precision=self.precision)

        cov = self._cov_cache[key]
        if cov.global_mask == 0:
            return cov, None

        ec_key = (cov.digest, M, N, K, str(device))
        if ec_key not in self._expanded_cache:
            self._expanded_cache[ec_key] = ExpandedCache(
                M, N, K, device, self.sa_rows, self.sa_cols)
        return cov, self._expanded_cache[ec_key]

    # ---- hook ------------------------------------------------------------

    def hook_fn(self, module, input_tuple, output_tuple):
        if not self.enabled or not self.fault_pe_row:
            return output_tuple

        X = input_tuple[0].contiguous()
        W = module.weight.T.contiguous()
        original_shape = X.shape
        X_2d = X.view(-1, original_shape[-1])
        device = W.device
        M, K = X_2d.shape
        _, N = W.shape

        cov, ec = self._ensure_coverage(device, M, N, K)

        if cov is None or cov.global_mask == 0:
            return output_tuple

        op = self.fault_config["op"]
        mode = self.fault_config["mode"]
        df = self.dataflow

        if df == "OS":
            Y = self._os_coverage(X_2d, W, M, N, K, device, cov, ec, op, mode)
        elif df == "WS":
            Y = self._ws_coverage(X_2d, W, M, N, K, device, cov, ec, op, mode)
        elif df == "IS":
            Y = self._is_coverage(X_2d, W, M, N, K, device, cov, ec, op, mode)
        else:
            return output_tuple

        Y = torch.clamp(Y, -1e3, 1e3)
        return Y.view(original_shape[:-1] + (N,))

    # ---- OS coverage -----------------------------------------------------

    def _os_coverage(self, X, W, M, N, K, device, cov, ec, op, mode):
        gm = cov.global_mask
        if mode == "input":
            Y = torch.matmul(X, W)
            X_corr = X.clone()
            mask_t = X_corr.new_full((1,), gm, dtype=torch.int32).expand(M, K)
            X_corr = self._inject_bit_error(X_corr, mask_t, op)
            dY = torch.matmul(X_corr, W) - Y

            S = cov.S_template.to(device)
            spatial = S[ec.i_mod.unsqueeze(1), ec.j_mod.unsqueeze(0)]
            Y += dY * spatial.to(Y.dtype)
            return Y

        elif mode == "weight":
            Y = torch.matmul(X, W)
            W_corr = W.clone()
            mask_t = W_corr.new_full((1,), gm, dtype=torch.int32).expand(K, N)
            W_corr = self._inject_bit_error(W_corr, mask_t, op)
            dY = torch.matmul(X, W_corr) - Y

            S = cov.S_template.to(device)
            spatial = S[ec.i_mod.unsqueeze(1), ec.j_mod.unsqueeze(0)]
            Y += dY * spatial.to(Y.dtype)
            return Y

        elif mode == "psum":
            return self._simulate_os(X, W, M, K, N, device)  # parent (already fast)

        return torch.matmul(X, W)

    # ---- WS coverage -----------------------------------------------------

    def _ws_coverage(self, X, W, M, N, K, device, cov, ec, op, mode):
        if mode == "weight":
            return self._simulate_ws(X, W, M, K, N, device)  # parent (already fast)

        if mode == "input":
            Y = torch.matmul(X, W)
            row_mask = cov.row_mask.to(device)       # [R]
            c_min = cov.c_min.to(device)              # [R]

            k_mask = row_mask[ec.k_mod]               # [K]
            active = k_mask != 0
            if not active.any():
                return Y

            X_corr = X.clone()
            mask_2d = k_mask.unsqueeze(0).expand(M, K)
            active_2d = active.unsqueeze(0).expand(M, K)
            X_corr[active_2d] = self._inject_bit_error(
                X_corr[active_2d], mask_2d[active_2d], op)
            dX = X_corr - X

            # H[k, j] = [j%C >= c_min[k%R]]
            c_min_k = c_min[ec.k_mod].unsqueeze(1)    # [K, 1]
            j_mod_2d = ec.j_mod.unsqueeze(0)           # [1, N]
            H = (j_mod_2d >= c_min_k).to(W.dtype)      # [K, N]

            Y += torch.matmul(dX, W * H)
            return Y

        elif mode == "psum":
            col_mask = cov.col_mask.to(device)         # [C] bool
            if not col_mask.any():
                return torch.matmul(X, W)

            Y = torch.matmul(X, W)
            j_active = col_mask[ec.j_mod].unsqueeze(0)  # [1, N]
            active_cols = j_active.expand(M, N)
            if not active_cols.any():
                return Y

            Y_corr = Y.clone()
            gm = cov.global_mask
            mask_t = Y_corr.new_full((1,), gm, dtype=torch.int32).expand(M, N)
            Y_corr[active_cols] = self._inject_bit_error(
                Y_corr[active_cols], mask_t[active_cols], op)
            Y += (Y_corr - Y)
            return Y

        return torch.matmul(X, W)

    # ---- IS coverage -----------------------------------------------------

    def _is_coverage(self, X, W, M, N, K, device, cov, ec, op, mode):
        if mode == "input":
            return self._simulate_is(X, W, M, K, N, device)  # parent (already fast)

        if mode == "weight":
            Y = torch.matmul(X, W)
            col_mask = cov.col_mask.to(device)         # [C]
            r_min = cov.r_min.to(device)               # [C]
            active_c = col_mask.nonzero(as_tuple=True)[0]
            if len(active_c) == 0:
                return Y

            gm = cov.global_mask
            W_corr = W.clone()
            mask_t = W_corr.new_full((1,), gm, dtype=torch.int32).expand(K, N)
            W_corr = self._inject_bit_error(W_corr, mask_t, op)
            dW = W_corr - W

            S = cov.S_template.to(device)
            X_weight = X * S[ec.i_mod.unsqueeze(1),
                              ec.k_mod.unsqueeze(0)].to(X.dtype)
            Y += torch.matmul(X_weight, dW)
            return Y

        elif mode == "psum":
            row_mask = cov.row_mask.to(device)         # [R] bool
            if not row_mask.any():
                return torch.matmul(X, W)

            Y = torch.matmul(X, W)
            i_active = row_mask[ec.i_mod].unsqueeze(1)  # [M, 1]
            active_rows = i_active.expand(M, N)
            if not active_rows.any():
                return Y

            Y_corr = Y.clone()
            gm = cov.global_mask
            mask_t = Y_corr.new_full((1,), gm, dtype=torch.int32).expand(M, N)
            Y_corr[active_rows] = self._inject_bit_error(
                Y_corr[active_rows], mask_t[active_rows], op)
            Y += (Y_corr - Y)
            return Y

        return torch.matmul(X, W)
