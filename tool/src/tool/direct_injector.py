"""
Direct bit-level fault injector — no SA hardware mapping baseline.

Injects bit errors directly into each linear layer's output tensor.
Unlike BER_Fast_SA_FaultInjector (PE grid + dataflow matmul), errors are
randomly distributed across the flat tensor with no spatial correlation.

When op="random", each faulty bit is independently stuck_0 or stuck_1 (50/50).

BER semantics: faults are sampled from the global bit space (all operators
combined). Each operator independently draws its share with probabilistic
rounding, so the expected total fault count equals total_bits × ber.
"""

import random

import torch


class DirectBitInjector:
    def __init__(self, op: str = "random", precision: str = "bf16"):
        self.op = op  # flip / stuck_0 / stuck_1 / random
        self.precision = precision.lower()
        self.enabled = True
        self._current_ber: float = 0.0
        self._sample_seed: int = 0

    def reset_for_sample(self, ber: float, seed: int):
        self._current_ber = ber
        self._sample_seed = seed

    # ------------------------------------------------------------------
    # fault positions: (bit_index, stuck_target) pairs
    # ------------------------------------------------------------------

    def _generate_positions(self, shape: torch.Size, op_seed: int,
                            num_bits: int = 32) -> list[tuple[int, int]]:
        total_elements = 1
        for d in shape:
            total_elements *= d
        total_bits = total_elements * num_bits
        expected = total_bits * self._current_ber
        num_faults = int(expected)

        # Probabilistic rounding: each operator independently carries the
        # fractional remainder, so expected total faults = global_total_bits × ber.
        rng = random.Random(op_seed)
        if rng.random() < (expected - num_faults):
            num_faults += 1

        if num_faults == 0:
            return []

        if num_faults > total_bits:
            num_faults = total_bits

        indices = rng.sample(range(total_bits), num_faults)

        if self.op == "random":
            return [(i, rng.choice([0, 1])) for i in indices]
        elif self.op == "stuck_0":
            return [(i, 0) for i in indices]
        elif self.op == "stuck_1":
            return [(i, 1) for i in indices]
        else:  # flip — use target=-1 as sentinel
            return [(i, -1) for i in indices]

    # ------------------------------------------------------------------
    # apply bit mask to tensor
    # ------------------------------------------------------------------

    def _apply_to(self, tensor: torch.Tensor, positions: list[tuple[int, int]]) -> torch.Tensor:
        if not positions:
            return tensor

        total_elements = tensor.numel()
        mask_0 = torch.zeros(total_elements, dtype=torch.int32, device=tensor.device)
        mask_1 = torch.zeros(total_elements, dtype=torch.int32, device=tensor.device)

        for pos, target in positions:
            elem_idx = pos // 32
            bit_idx = pos % 32
            if elem_idx < total_elements:
                bit = 1 << bit_idx
                if target == -1:
                    mask_1[elem_idx] ^= bit  # flip via XOR later with combined mask
                elif target == 0:
                    mask_0[elem_idx] |= bit
                else:
                    mask_1[elem_idx] |= bit

        orig_dtype = tensor.dtype
        # Flatten to 1D for mask application, then restore shape
        shape_2d = tensor.shape
        t_int = tensor.float().view(torch.int32).view(-1)

        if self.op == "flip":
            t_int = t_int ^ mask_1
        elif self.op == "random":
            t_int = t_int & (~mask_0)  # clear stuck_0 bits
            t_int = t_int | mask_1     # set stuck_1 bits
        elif self.op == "stuck_0":
            t_int = t_int & (~mask_0)
        elif self.op == "stuck_1":
            t_int = t_int | mask_1

        t_int = t_int.view(shape_2d)

        out = t_int.view(torch.float32)
        out = torch.nan_to_num(out, nan=3.4e38, posinf=3.4e38, neginf=-3.4e38)
        return out.to(orig_dtype)

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def make_hook(self, op_id: str):
        """Create a forward hook for a specific operator.

        Each operator gets a unique seed derived from op_id, ensuring
        independent fault sampling across all operators in the global
        bit space.
        """
        def hook_fn(module, input_tuple, output_tuple):
            if not self.enabled:
                return output_tuple
            op_seed = self._sample_seed + hash(op_id)
            positions = self._generate_positions(output_tuple.shape, op_seed)
            return self._apply_to(output_tuple, positions)
        return hook_fn

    def hook_fn(self, module, input_tuple, output_tuple):
        """Legacy hook — uses output shape hash as seed (non-unique across layers).
        Prefer make_hook(op_id) for proper global BER semantics.
        """
        if not self.enabled:
            return output_tuple
        op_seed = self._sample_seed + hash(output_tuple.shape)
        positions = self._generate_positions(output_tuple.shape, op_seed)
        return self._apply_to(output_tuple, positions)
