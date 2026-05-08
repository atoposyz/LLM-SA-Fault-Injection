import json
import math
import os

import torch


def extract_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            tensor = extract_tensor(item)
            if tensor is not None:
                return tensor
    if hasattr(obj, "last_hidden_state") and torch.is_tensor(obj.last_hidden_state):
        return obj.last_hidden_state
    return None


def prepare_metric_matrix(tensor: torch.Tensor):
    if tensor is None:
        return None

    matrix = tensor.detach()
    if matrix.numel() == 0:
        return None

    if matrix.dtype in {torch.bfloat16, torch.float16}:
        matrix = matrix.float()
    elif not torch.is_floating_point(matrix):
        matrix = matrix.float()

    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1e4, neginf=-1e4)

    if matrix.ndim == 0:
        matrix = matrix.view(1, 1)
    elif matrix.ndim == 1:
        matrix = matrix.view(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])

    return matrix


def compute_runtime_metrics(tensor: torch.Tensor):
    matrix = prepare_metric_matrix(tensor)
    if matrix is None:
        return None

    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return None

    try:
        singular_values = torch.linalg.svdvals(matrix)
    except RuntimeError:
        singular_values = torch.linalg.svdvals(matrix.cpu())

    singular_values = singular_values.float().cpu()
    if singular_values.numel() == 0:
        return None

    energy = singular_values.square()
    frobenius_norm_sq = float(energy.sum().item())
    spectral_norm_sq = float(energy.max().item()) if energy.numel() > 0 else 0.0
    spectral_norm = math.sqrt(spectral_norm_sq) if spectral_norm_sq > 0.0 else 0.0
    stable_rank = frobenius_norm_sq / spectral_norm_sq if spectral_norm_sq > 0.0 else 0.0

    positive_mask = energy > 0
    if frobenius_norm_sq > 0.0 and positive_mask.any():
        probabilities = energy[positive_mask] / frobenius_norm_sq
        svd_entropy = float((-(probabilities * torch.log(probabilities))).sum().item())
    else:
        svd_entropy = 0.0

    energy_sq_sum = float(energy.square().sum().item()) if energy.numel() > 0 else 0.0
    participation_ratio = (
        (frobenius_norm_sq * frobenius_norm_sq) / energy_sq_sum if energy_sq_sum > 0.0 else 0.0
    )

    topk = min(5, energy.numel())
    top5_energy = float(energy.topk(topk).values.sum().item()) if topk > 0 else 0.0
    top5_energy_ratio = top5_energy / frobenius_norm_sq if frobenius_norm_sq > 0.0 else 0.0

    sigma_max = float(singular_values.max().item()) if singular_values.numel() > 0 else 0.0
    tol = max(rows, cols) * torch.finfo(torch.float32).eps * sigma_max
    numerical_rank = int((singular_values > tol).sum().item()) if sigma_max > 0.0 else 0

    nuclear_norm = float(singular_values.sum().item())
    nuclear_rank = nuclear_norm / spectral_norm if spectral_norm > 0.0 else 0.0
    normalized_nuclear_rank = nuclear_rank / max(1, min(rows, cols))

    return {
        "matrix_rows": int(rows),
        "matrix_cols": int(cols),
        "stable_rank": float(stable_rank),
        "svd_entropy": float(svd_entropy),
        "participation_ratio": float(participation_ratio),
        "top5_energy_ratio": float(top5_energy_ratio),
        "frobenius_norm_sq": float(frobenius_norm_sq),
        "spectral_norm_sq": float(spectral_norm_sq),
        "numerical_rank": int(numerical_rank),
        "nuclear_norm": float(nuclear_norm),
        "normalized_nuclear_rank": float(normalized_nuclear_rank),
        "nuclear_rank": float(nuclear_rank),
    }


class RuntimeMetricsWriter:
    def __init__(self, output_path=None, interval=20, run_tag="default", extra_meta=None, flush_every=1):
        self.output_path = output_path
        self.interval = max(1, int(interval))
        self.run_tag = run_tag
        self.extra_meta = dict(extra_meta or {})
        self.flush_every = max(1, int(flush_every))
        self.enabled = output_path is not None
        self.buffer = []
        self.global_step = 0
        self.module_counters = {}

        if self.output_path:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

    def reset(self):
        self.buffer = []
        self.global_step = 0
        self.module_counters = {}

    def reconfigure(
        self,
        output_path=None,
        interval=20,
        run_tag="default",
        enabled=True,
        extra_meta=None,
        flush_every=1,
        reset_counters=True,
    ):
        self.output_path = output_path
        self.interval = max(1, int(interval))
        self.run_tag = run_tag
        self.enabled = enabled and output_path is not None
        self.extra_meta = dict(extra_meta or {})
        self.flush_every = max(1, int(flush_every))

        if self.output_path:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        if reset_counters:
            self.reset()

    def set_run_tag(self, run_tag, extra_meta=None, reset_counters=True):
        self.run_tag = run_tag
        if extra_meta is not None:
            self.extra_meta = dict(extra_meta)
        if reset_counters:
            self.reset()

    def next_step(self, module_name):
        self.global_step += 1
        module_step = self.module_counters.get(module_name, 0) + 1
        self.module_counters[module_name] = module_step
        should_capture = module_step == 1 or (module_step % self.interval == 0)
        return should_capture, module_step, self.global_step

    def write_record(self, record):
        if not self.enabled or not self.output_path:
            return
        self.buffer.append(record)
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buffer or not self.output_path:
            return
        with open(self.output_path, "a", encoding="utf-8") as handle:
            for record in self.buffer:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        self.buffer = []
