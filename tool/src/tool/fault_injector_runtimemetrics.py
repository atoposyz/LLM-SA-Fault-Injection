import os
import random

import torch
import torch.nn as nn

from .runtime_metrics import RuntimeMetricsWriter, compute_runtime_metrics, extract_tensor


class Fast_RuntimeMetrics_SA_FaultInjector:
    """
    High-performance GPU Systolic Array Fault Injector using Tensor Masking.
    Supports Dataflows: WS (Weight Stationary), OS (Output Stationary), IS (Input Stationary)
    Supports Faults: input, weight, psum
    Supports Types: bitflip, stuck_0, stuck_1

    Extra runtime-metric support:
    - Stable Rank
    - SVD Entropy
    - Participation Ratio
    - Top-5 Energy Ratio
    - Frobenius Norm Squared
    - Spectral Norm Squared
    - Numerical Rank
    - Nuclear Norm
    - Normalized Nuclear Rank
    """

    def __init__(
        self,
        sa_rows=256,
        sa_cols=256,
        dataflow="WS",
        fault_type="input_bitflip_12",
        precision="bf16",
        metric_interval=20,
    ):
        self.sa_rows = sa_rows
        self.sa_cols = sa_cols
        self.dataflow = dataflow.upper()
        self.fault_type_str = fault_type.lower()
        self.precision = precision.lower()

        self.enabled = True
        self.fault_pe_row = []
        self.fault_pe_col = []
        self.fault_reg = []
        self.fault_bit = []
        self._pe_mask_cache = {}

        self.metric_interval = max(1, int(metric_interval))
        self.metric_enabled = False
        self.metric_output_path = None
        self.metric_writer = RuntimeMetricsWriter(interval=self.metric_interval)

        self.parse_fault_type()

    def print_config(self):
        print("Fast SA Fault Injector Configuration (Tensor Masking):")
        print(f"  SA Size: {self.sa_rows} rows x {self.sa_cols} cols")
        print(f"  Dataflow: {self.dataflow}")
        print(f"  Fault Type: {self.fault_type_str}")
        print(f"  Parsed Config: {self.fault_config}")
        print(f"  Runtime Metric Interval: {self.metric_interval}")
        print(f"  Runtime Metric Enabled: {self.metric_enabled}")
        print(f"  Runtime Metric Output: {self.metric_output_path}")
        if hasattr(self, "fault_reg") and len(self.fault_reg) == len(self.fault_pe_row):
            print(
                "  Fault Details (row, col, reg, bit): "
                f"{list(zip(self.fault_pe_row, self.fault_pe_col, self.fault_reg, self.fault_bit))}"
            )
        else:
            print(f"  Fault PE Position: {list(zip(self.fault_pe_row, self.fault_pe_col))}")
        print(f"  Enabled: {self.enabled}")

    def configure_runtime_metrics(
        self,
        output_path,
        interval=20,
        run_tag="default",
        enabled=True,
        extra_meta=None,
        flush_every=1,
        reset_counters=True,
    ):
        self.metric_output_path = output_path
        self.metric_interval = max(1, int(interval))
        self.metric_enabled = enabled
        self.metric_writer.reconfigure(
            output_path=output_path,
            interval=interval,
            run_tag=run_tag,
            enabled=enabled,
            extra_meta=extra_meta,
            flush_every=flush_every,
            reset_counters=reset_counters,
        )

    def reset_runtime_metric_state(self):
        self.metric_writer.reset()

    def set_runtime_metric_run_tag(self, run_tag, extra_meta=None, reset_counters=True):
        self.metric_writer.set_run_tag(run_tag, extra_meta=extra_meta, reset_counters=reset_counters)

    def flush_runtime_metrics(self):
        self.metric_writer.flush()

    def register_metric_hook(self, module: nn.Module, layer_name=None):
        if layer_name:
            setattr(module, "_runtime_metrics_name", layer_name)
        return module.register_forward_hook(self.hook_fn)

    def parse_fault_type(self):
        parts = self.fault_type_str.split("_")
        mode = parts[0]
        f_type = parts[1]

        op = "flip"
        pos = 0
        if f_type == "bitflip":
            op = "flip"
            pos = int(parts[2]) if len(parts) > 2 else 0
        elif f_type == "stuck":
            val = parts[2]
            op = f"stuck_{val}"
            pos = int(parts[3]) if len(parts) > 3 else 0

        self.fault_config = {"mode": mode, "op": op, "pos": pos}

    def set_fault_config_injpos(self, val):
        self.fault_config["pos"] = val

    def reset_fault_pe(self):
        self.fault_pe_row = []
        self.fault_pe_col = []
        self.fault_reg = []
        self.fault_bit = []
        self._pe_mask_cache = {}

    def init_fault_position(self):
        self.reset_fault_pe()
        self.fault_pe_row.append(random.randint(0, self.sa_rows - 1))
        self.fault_pe_col.append(random.randint(0, self.sa_cols - 1))
        self.fault_reg.append(0)
        self.fault_bit.append(self.fault_config.get("pos", 0))

    def init_multi_fault_positions(self, num_faults: int, num_regs: int = 1, num_bits: int = 16):
        self.reset_fault_pe()
        max_pe_positions = self.sa_rows * self.sa_cols
        max_positions = max_pe_positions * num_regs * num_bits
        if num_faults <= 0:
            return
        if num_faults > max_positions:
            raise ValueError(f"num_faults ({num_faults}) exceeds total bit count ({max_positions})")

        flat_indices = random.sample(range(max_positions), num_faults)
        for idx in flat_indices:
            pe_idx = idx // (num_regs * num_bits)
            remainder = idx % (num_regs * num_bits)
            reg_idx = remainder // num_bits
            bit_idx = remainder % num_bits

            self.fault_pe_row.append(pe_idx // self.sa_cols)
            self.fault_pe_col.append(pe_idx % self.sa_cols)
            self.fault_reg.append(reg_idx)
            self.fault_bit.append(bit_idx)

    def set_fault_position(self, row: int, col: int):
        self.reset_fault_pe()
        if not (0 <= row < self.sa_rows and 0 <= col < self.sa_cols):
            raise ValueError("PE bounds exceeded")
        self.fault_pe_row.append(row)
        self.fault_pe_col.append(col)
        self.fault_reg.append(0)
        self.fault_bit.append(self.fault_config.get("pos", 0))

    def set_multi_fault_positions(self, positions: list):
        self.reset_fault_pe()
        for row, col in positions:
            if not (0 <= row < self.sa_rows and 0 <= col < self.sa_cols):
                raise ValueError("PE bounds exceeded")
            self.fault_pe_row.append(row)
            self.fault_pe_col.append(col)
            self.fault_reg.append(0)
            self.fault_bit.append(self.fault_config.get("pos", 0))

    def _inject_bit_error(self, tensor: torch.Tensor, mask: torch.Tensor, op: str):
        if tensor.numel() == 0:
            return tensor

        orig_dtype = tensor.dtype
        tensor_fp32 = tensor.float() if orig_dtype != torch.float32 else tensor
        t_int = tensor_fp32.view(torch.int32)

        if op == "stuck_1":
            t_int = t_int | mask
        elif op == "stuck_0":
            t_int = t_int & (~mask)
        elif op == "flip":
            t_int = t_int ^ mask

        out = t_int.view(torch.float32)
        out = torch.nan_to_num(out, nan=3.4e38, posinf=3.4e38, neginf=-3.4e38)

        return out.to(orig_dtype)

    def _build_pe_mask_map(self, device):
        fingerprint = (
            tuple(self.fault_pe_row),
            tuple(self.fault_pe_col),
            tuple(self.fault_reg),
            tuple(self.fault_bit),
            self.fault_config["op"],
            self.precision,
        )
        device_key = str(device)
        cached = self._pe_mask_cache.get(device_key)
        if cached is not None and cached[1] == fingerprint:
            return cached[0]

        pe_mask_map_cpu = torch.zeros((self.sa_rows, self.sa_cols), dtype=torch.int32)
        op = self.fault_config["op"]

        for f_idx in range(len(self.fault_pe_row)):
            r = self.fault_pe_row[f_idx]
            c = self.fault_pe_col[f_idx]
            pos = self.fault_bit[f_idx]

            mask_pos = pos + 16 if self.precision == "bf16" and pos < 16 else pos
            bit = 1 << mask_pos

            if op == "flip":
                pe_mask_map_cpu[r, c] ^= bit
            elif op in {"stuck_1", "stuck_0"}:
                pe_mask_map_cpu[r, c] |= bit

        result = pe_mask_map_cpu.to(device)
        self._pe_mask_cache[device_key] = (result, fingerprint)
        return result

    def _next_metric_state(self, module_name):
        return self.metric_writer.next_step(module_name)

    def _record_runtime_metrics(self, module, tensor, stage, module_step, global_step):
        if not self.metric_enabled or not self.metric_output_path:
            return

        metrics = compute_runtime_metrics(tensor)
        if metrics is None:
            return

        module_name = getattr(module, "_runtime_metrics_name", None)
        if module_name is None:
            module_name = f"{module.__class__.__name__}_{id(module)}"

        record = {
            "run_tag": self.metric_writer.run_tag,
            "layer_name": module_name,
            "module_type": module.__class__.__name__,
            "stage": stage,
            "module_step": int(module_step),
            "global_step": int(global_step),
            "metric_interval": int(self.metric_interval),
            "fault_enabled": bool(self.enabled and bool(self.fault_pe_row)),
            "fault_type": self.fault_type_str,
            "dataflow": self.dataflow,
            "fault_config": dict(self.fault_config),
            "fault_pe_row": list(self.fault_pe_row),
            "fault_pe_col": list(self.fault_pe_col),
            "fault_reg": list(self.fault_reg),
            "fault_bit": list(self.fault_bit),
        }
        record.update(self.metric_writer.extra_meta)
        record.update(metrics)

        self.metric_writer.write_record(record)

    def hook_fn(self, module, input_tuple, output_tuple):
        clean_output = extract_tensor(output_tuple)
        module_name = getattr(module, "_runtime_metrics_name", f"{module.__class__.__name__}_{id(module)}")
        should_capture, module_step, global_step = self._next_metric_state(module_name)

        if should_capture and clean_output is not None:
            clean_stage = "clean_output" if self.enabled and self.fault_pe_row else "output"
            self._record_runtime_metrics(module, clean_output, clean_stage, module_step, global_step)

        if not self.enabled or not self.fault_pe_row:
            return output_tuple

        X = input_tuple[0].contiguous()
        W = module.weight.T.contiguous()
        original_shape = X.shape
        X_2d = X.view(-1, original_shape[-1])

        device = W.device
        M, K = X_2d.shape
        _, N = W.shape

        r_f = torch.tensor(self.fault_pe_row, device=device).view(-1, 1, 1)
        c_f = torch.tensor(self.fault_pe_col, device=device).view(-1, 1, 1)

        if self.dataflow == "WS":
            Y_faulty = self._simulate_ws(X_2d, W, M, K, N, r_f, c_f, device)
        elif self.dataflow == "OS":
            Y_faulty = self._simulate_os(X_2d, W, M, K, N, r_f, c_f, device)
        elif self.dataflow == "IS":
            Y_faulty = self._simulate_is(X_2d, W, M, K, N, r_f, c_f, device)
        else:
            raise ValueError(f"Unknown dataflow: {self.dataflow}")

        Y_faulty = torch.clamp(Y_faulty, -1e3, 1e3)
        faulty_output = Y_faulty.view(original_shape[:-1] + (N,))

        if should_capture:
            self._record_runtime_metrics(module, faulty_output, "faulty_output", module_step, global_step)

        return faulty_output

    def _simulate_ws(self, X, W, M, K, N, r_f, c_f, device):
        k_mod = (torch.arange(K, device=device) % self.sa_rows).unsqueeze(1)
        j_mod = (torch.arange(N, device=device) % self.sa_cols).unsqueeze(0)

        mode = self.fault_config["mode"]
        op = self.fault_config["op"]
        pe_mask_map = self._build_pe_mask_map(device)

        if mode == "weight":
            W_tilde = W.clone()
            mapped_mask = pe_mask_map[k_mod.expand(K, N), j_mod.expand(K, N)]
            active_mask = mapped_mask > 0 if op != "stuck_0" else mapped_mask != 0
            if active_mask.any():
                W_tilde[active_mask] = self._inject_bit_error(W_tilde[active_mask], mapped_mask[active_mask], op)
            return torch.matmul(X, W_tilde)

        if mode == "input":
            Y_tilde = torch.matmul(X, W)
            active_cols = torch.nonzero(pe_mask_map.sum(dim=0)).squeeze(-1)
            if active_cols.dim() == 0:
                active_cols = active_cols.unsqueeze(0)

            for c in active_cols:
                col_mask = pe_mask_map[:, c]
                k_mask_for_c = col_mask[k_mod.squeeze(-1)]

                active_k = k_mask_for_c > 0 if op != "stuck_0" else k_mask_for_c != 0
                if not active_k.any():
                    continue

                X_flipped = X.clone()
                mask_2d = k_mask_for_c.unsqueeze(0).expand(M, K)
                active_2d = active_k.unsqueeze(0).expand(M, K)
                X_flipped[active_2d] = self._inject_bit_error(X_flipped[active_2d], mask_2d[active_2d], op)

                dx = X_flipped - X
                in_mask = (j_mod >= c).squeeze(0)
                W_masked = W * in_mask.to(W.dtype)
                Y_tilde += torch.matmul(dx, W_masked)

            return Y_tilde

        if mode == "psum":
            Y = torch.matmul(X, W)
            Y_tilde = Y.clone()

            active_rc = torch.nonzero(pe_mask_map)
            for point in active_rc:
                r, c = point[0].item(), point[1].item()
                mask_val = pe_mask_map[r, c]
                affected_j = torch.where((torch.arange(N, device=device) % self.sa_cols) == c)[0]

                scale = max((K - r) / K, 0.0) if K > 0 else 1.0
                if affected_j.numel() > 0:
                    col_data = Y[:, affected_j]
                    mask_tensor = mask_val.expand(col_data.shape)
                    flipped = self._inject_bit_error(col_data, mask_tensor, op)
                    diff = flipped - col_data
                    Y_tilde[:, affected_j] += diff * scale

            return Y_tilde

        return torch.matmul(X, W)

    def _simulate_os(self, X, W, M, K, N, r_f, c_f, device):
        i_mod = (torch.arange(M, device=device) % self.sa_rows).unsqueeze(1)
        j_mod = (torch.arange(N, device=device) % self.sa_cols).unsqueeze(0)

        Y = torch.matmul(X, W)
        mode = self.fault_config["mode"]
        op = self.fault_config["op"]
        pe_mask_map = self._build_pe_mask_map(device)

        if mode == "psum":
            Y_tilde = Y.clone()
            mapped_mask = pe_mask_map[i_mod.expand(M, N), j_mod.expand(M, N)]
            active_mask = mapped_mask > 0 if op != "stuck_0" else mapped_mask != 0
            if active_mask.any():
                Y_tilde[active_mask] = self._inject_bit_error(Y_tilde[active_mask], mapped_mask[active_mask], op)
            return Y_tilde

        if mode == "input":
            Y_tilde = Y.clone()
            active_rc = torch.nonzero(pe_mask_map)
            for point in active_rc:
                r, c = point[0].item(), point[1].item()
                mask_val = pe_mask_map[r, c]
                in_mask = (i_mod == r) & (j_mod >= c)

                mask_tensor = mask_val.expand(X.shape)
                X_flipped = self._inject_bit_error(X, mask_tensor, op)
                diff = torch.matmul(X_flipped, W) - Y
                Y_tilde += diff * in_mask.to(Y_tilde.dtype)

            return Y_tilde

        if mode == "weight":
            Y_tilde = Y.clone()
            active_rc = torch.nonzero(pe_mask_map)
            for point in active_rc:
                r, c = point[0].item(), point[1].item()
                mask_val = pe_mask_map[r, c]
                wt_mask = (i_mod >= r) & (j_mod == c)

                mask_tensor = mask_val.expand(W.shape)
                W_flipped = self._inject_bit_error(W, mask_tensor, op)
                diff = torch.matmul(X, W_flipped) - Y
                Y_tilde += diff * wt_mask.to(Y_tilde.dtype)

            return Y_tilde

        return Y

    def _simulate_is(self, X, W, M, K, N, r_f, c_f, device):
        i_mod = (torch.arange(M, device=device) % self.sa_rows).unsqueeze(1)
        k_mod = (torch.arange(K, device=device) % self.sa_cols).unsqueeze(0)

        Y = torch.matmul(X, W)
        mode = self.fault_config["mode"]
        op = self.fault_config["op"]
        pe_mask_map = self._build_pe_mask_map(device)

        if mode == "input":
            X_tilde = X.clone()
            mapped_mask = pe_mask_map[i_mod.expand(M, K), k_mod.expand(M, K)]
            active_mask = mapped_mask > 0 if op != "stuck_0" else mapped_mask != 0
            if active_mask.any():
                X_tilde[active_mask] = self._inject_bit_error(X_tilde[active_mask], mapped_mask[active_mask], op)
            return torch.matmul(X_tilde, W)

        if mode == "weight":
            Y_tilde = Y.clone()
            active_rc = torch.nonzero(pe_mask_map)
            for point in active_rc:
                r, c = point[0].item(), point[1].item()
                mask_val = pe_mask_map[r, c]
                wt_mask = (i_mod >= r) & (k_mod == c)

                mask_tensor = mask_val.expand(W.shape)
                W_flipped = self._inject_bit_error(W, mask_tensor, op)
                W_diff = W_flipped - W
                X_masked = X * wt_mask.to(X.dtype)
                Y_tilde += torch.matmul(X_masked, W_diff)

            return Y_tilde

        if mode == "psum":
            Y_tilde = Y.clone()
            active_rc = torch.nonzero(pe_mask_map)
            for point in active_rc:
                r, c = point[0].item(), point[1].item()
                mask_val = pe_mask_map[r, c]
                affected_i = torch.where((torch.arange(M, device=device) % self.sa_rows) == r)[0]

                scale = max((K - c) / K, 0.0) if K > 0 else 1.0
                if affected_i.numel() > 0:
                    row_data = Y[affected_i, :]
                    mask_tensor = mask_val.expand(row_data.shape)
                    flipped = self._inject_bit_error(row_data, mask_tensor, op)
                    diff = flipped - row_data
                    Y_tilde[affected_i, :] += diff * scale

            return Y_tilde

        return Y
