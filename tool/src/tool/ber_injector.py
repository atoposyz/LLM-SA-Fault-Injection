import torch
import torch.nn as nn
import random

class BER_Fast_SA_FaultInjector:
    """
    针对误码率（BER）和多比特的脉动阵列故障注入器。
    专为大规模、高并发的随机多点注入优化，移除所有单比特冗余逻辑。
    """
    def __init__(self, sa_rows=256, sa_cols=256, dataflow='WS', fault_type='random_bitflip_mixed', precision='bf16'):
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
        
        self.parse_fault_type()

    def print_config(self):
        print(f"BER Fast SA Fault Injector:")
        print(f"  SA Size: {self.sa_rows}x{self.sa_cols} | Dataflow: {self.dataflow}")
        print(f"  Fault Type: {self.fault_config['mode']}_{self.fault_config['op']} | Enabled: {self.enabled}")
        print(f"  Current Total Injected Faults: {len(self.fault_pe_row)}")

    def parse_fault_type(self):
        parts = self.fault_type_str.split('_')
        mode = parts[0]
        f_type = parts[1]
        
        if f_type == 'bitflip':
            op = 'flip'
        elif f_type == 'stuck':
            val = parts[2]
            op = f'stuck_{val}'
        else:
            op = 'flip'
            
        self.fault_config = {'mode': mode, 'op': op}

    def reset_fault_pe(self):
        self.fault_pe_row.clear()
        self.fault_pe_col.clear()
        self.fault_reg.clear()
        self.fault_bit.clear()
        self._pe_mask_cache.clear()

    def init_faults_by_ber(self, ber: float, num_regs: int = 1, num_bits: int = 16):
        """直接通过误码率(BER)全局撒错"""
        max_pe_positions = self.sa_rows * self.sa_cols
        max_positions = max_pe_positions * num_regs * num_bits
        num_faults = int(max_positions * ber)
        
        print(f"[BER Injector] Total Bits: {max_positions}, BER: {ber}, Injecting {num_faults} faults.")
        self.init_multi_fault_positions(num_faults, num_regs, num_bits)

    def init_multi_fault_positions(self, num_faults: int, num_regs: int = 1, num_bits: int = 16):
        """通过指定的错误数量(Num_faults)全局无放回随机撒错"""
        self.reset_fault_pe()
        max_pe_positions = self.sa_rows * self.sa_cols
        max_positions = max_pe_positions * num_regs * num_bits
        if num_faults <= 0: return
        if num_faults > max_positions:
            raise ValueError(f"num_faults ({num_faults}) exceeds total bit count ({max_positions})")
        
        # 核心：无放回随机抽样保证同一个 bit 绝对不会被重复注入
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

    # =====================================================================
    # 核心模拟与 Mask 逻辑
    # =====================================================================
    def _inject_bit_error(self, tensor: torch.Tensor, mask: torch.Tensor, op: str):
        if tensor.numel() == 0: return tensor
        orig_dtype = tensor.dtype
        tensor_fp32 = tensor.float() if orig_dtype != torch.float32 else tensor
        t_int = tensor_fp32.view(torch.int32)
        
        if op == 'stuck_1': t_int = t_int | mask
        elif op == 'stuck_0': t_int = t_int & (~mask)
        elif op == 'flip': t_int = t_int ^ mask
        
        out = t_int.view(torch.float32)
        out = torch.nan_to_num(out, nan=3.4e38, posinf=3.4e38, neginf=-3.4e38)
        return out.to(orig_dtype)

    def _build_pe_mask_map(self, device):
        fingerprint = len(self.fault_pe_row)
        device_key = str(device)
        cached = self._pe_mask_cache.get(device_key)
        if cached is not None and cached[1] == fingerprint:
            return cached[0]
        
        pe_mask_map_cpu = torch.zeros((self.sa_rows, self.sa_cols), dtype=torch.int32)
        op = self.fault_config['op']
        
        for f_idx in range(len(self.fault_pe_row)):
            r, c, pos = self.fault_pe_row[f_idx], self.fault_pe_col[f_idx], self.fault_bit[f_idx]
            mask_pos = pos + 16 if self.precision == 'bf16' and pos < 16 else pos
            bit = 1 << mask_pos
            if op == 'flip': pe_mask_map_cpu[r, c] ^= bit
            elif op in ['stuck_1', 'stuck_0']: pe_mask_map_cpu[r, c] |= bit
                
        result = pe_mask_map_cpu.to(device)
        self._pe_mask_cache[device_key] = (result, fingerprint)
        return result

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
        
        if self.dataflow == 'WS': Y_faulty = self._simulate_ws(X_2d, W, M, K, N, device)
        elif self.dataflow == 'OS': Y_faulty = self._simulate_os(X_2d, W, M, K, N, device)
        elif self.dataflow == 'IS': Y_faulty = self._simulate_is(X_2d, W, M, K, N, device)
        else: raise ValueError(f"Unknown dataflow: {self.dataflow}")
            
        Y_faulty = torch.clamp(Y_faulty, -1e3, 1e3)
        return Y_faulty.view(original_shape[:-1] + (N,))

    def _simulate_ws(self, X, W, M, K, N, device):
        k_mod = (torch.arange(K, device=device) % self.sa_rows).unsqueeze(1) 
        j_mod = (torch.arange(N, device=device) % self.sa_cols).unsqueeze(0) 
        mode, op = self.fault_config['mode'], self.fault_config['op']
        pe_mask_map = self._build_pe_mask_map(device) 
        
        if mode == 'weight':
            W_tilde = W.clone()
            mapped_mask = pe_mask_map[k_mod.expand(K, N), j_mod.expand(K, N)]
            active_mask = mapped_mask != 0 
            if active_mask.any(): W_tilde[active_mask] = self._inject_bit_error(W_tilde[active_mask], mapped_mask[active_mask], op)
            return torch.matmul(X, W_tilde)
        elif mode == 'input':
            Y_tilde = torch.matmul(X, W)
            active_cols = torch.nonzero(pe_mask_map.sum(dim=0)).squeeze(-1)
            if active_cols.dim() == 0: active_cols = active_cols.unsqueeze(0)
            for c in active_cols:
                col_mask = pe_mask_map[:, c] 
                k_mask_for_c = col_mask[k_mod.squeeze(-1)] 
                active_k = k_mask_for_c != 0
                if not active_k.any(): continue
                X_flipped = X.clone()
                mask_2d = k_mask_for_c.unsqueeze(0).expand(M, K)
                active_2d = active_k.unsqueeze(0).expand(M, K)
                X_flipped[active_2d] = self._inject_bit_error(X_flipped[active_2d], mask_2d[active_2d], op)
                dx = X_flipped - X 
                in_mask = (j_mod >= c).squeeze(0) 
                W_masked = W * in_mask.to(W.dtype) 
                Y_tilde += torch.matmul(dx, W_masked)
            return Y_tilde
        elif mode == 'psum':
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
                    Y_tilde[:, affected_j] += (flipped - col_data) * scale
            return Y_tilde
        return torch.matmul(X, W)

    def _simulate_os(self, X, W, M, K, N, device):
        i_mod = (torch.arange(M, device=device) % self.sa_rows).unsqueeze(1) 
        j_mod = (torch.arange(N, device=device) % self.sa_cols).unsqueeze(0) 
        Y = torch.matmul(X, W)
        mode, op = self.fault_config['mode'], self.fault_config['op']
        pe_mask_map = self._build_pe_mask_map(device)
        
        if mode == 'psum':
            Y_tilde = Y.clone()
            mapped_mask = pe_mask_map[i_mod.expand(M, N), j_mod.expand(M, N)]
            active_mask = mapped_mask != 0
            if active_mask.any(): Y_tilde[active_mask] = self._inject_bit_error(Y_tilde[active_mask], mapped_mask[active_mask], op)
            return Y_tilde
        elif mode == 'input':
            Y_tilde = Y.clone()
            for point in torch.nonzero(pe_mask_map):
                r, c = point[0].item(), point[1].item()
                in_mask = (i_mod == r) & (j_mod >= c)
                X_flipped = self._inject_bit_error(X, pe_mask_map[r, c].expand(X.shape), op)
                Y_tilde += (torch.matmul(X_flipped, W) - Y) * in_mask.to(Y_tilde.dtype)
            return Y_tilde
        elif mode == 'weight':
            Y_tilde = Y.clone()
            for point in torch.nonzero(pe_mask_map):
                r, c = point[0].item(), point[1].item()
                wt_mask = (i_mod >= r) & (j_mod == c)
                W_flipped = self._inject_bit_error(W, pe_mask_map[r, c].expand(W.shape), op)
                Y_tilde += (torch.matmul(X, W_flipped) - Y) * wt_mask.to(Y_tilde.dtype)
            return Y_tilde
        return Y

    def _simulate_is(self, X, W, M, K, N, device):
        i_mod = (torch.arange(M, device=device) % self.sa_rows).unsqueeze(1) 
        k_mod = (torch.arange(K, device=device) % self.sa_cols).unsqueeze(0) 
        Y = torch.matmul(X, W)
        mode, op = self.fault_config['mode'], self.fault_config['op']
        pe_mask_map = self._build_pe_mask_map(device)
        
        if mode == 'input':
            X_tilde = X.clone()
            mapped_mask = pe_mask_map[i_mod.expand(M, K), k_mod.expand(M, K)]
            active_mask = mapped_mask != 0
            if active_mask.any(): X_tilde[active_mask] = self._inject_bit_error(X_tilde[active_mask], mapped_mask[active_mask], op)
            return torch.matmul(X_tilde, W)
        elif mode == 'weight':
            Y_tilde = Y.clone()
            for point in torch.nonzero(pe_mask_map):
                r, c = point[0].item(), point[1].item()
                wt_mask = (i_mod >= r) & (k_mod == c)
                W_flipped = self._inject_bit_error(W, pe_mask_map[r, c].expand(W.shape), op)
                X_masked = X * wt_mask.to(X.dtype)
                Y_tilde += torch.matmul(X_masked, W_flipped - W)
            return Y_tilde
        elif mode == 'psum':
            Y_tilde = Y.clone()
            for point in torch.nonzero(pe_mask_map):
                r, c = point[0].item(), point[1].item()
                affected_i = torch.where((torch.arange(M, device=device) % self.sa_rows) == r)[0]
                scale = max((K - c) / K, 0.0) if K > 0 else 1.0
                if affected_i.numel() > 0:
                    row_data = Y[affected_i, :]
                    flipped = self._inject_bit_error(row_data, pe_mask_map[r, c].expand(row_data.shape), op)
                    Y_tilde[affected_i, :] += (flipped - row_data) * scale
            return Y_tilde
        return Y