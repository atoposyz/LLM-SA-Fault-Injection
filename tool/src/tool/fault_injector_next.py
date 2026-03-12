import torch
import torch.nn as nn
import random
import math

class Fast_SA_FaultInjector:
    """
    High-performance GPU Systolic Array Fault Injector using Tensor Masking.
    Supports Dataflows: WS (Weight Stationary), OS (Output Stationary), IS (Input Stationary)
    Supports Faults: input, weight, psum 
    Supports Types: bitflip, stuck_0, stuck_1
    """
    def __init__(self, sa_rows=256, sa_cols=256, dataflow='WS', fault_type='input_bitflip_12', precision='bf16'):
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
        self._pe_mask_cache = {}  # device -> (tensor, fault_fingerprint)
        
        self.parse_fault_type()

    def print_config(self):
        print(f"Fast SA Fault Injector Configuration (Tensor Masking):")
        print(f"  SA Size: {self.sa_rows} rows x {self.sa_cols} cols")
        print(f"  Dataflow: {self.dataflow}")
        print(f"  Fault Type: {self.fault_type_str}")
        print(f"  Parsed Config: {self.fault_config}")
        if hasattr(self, 'fault_reg') and len(self.fault_reg) == len(self.fault_pe_row):
            print(f"  Fault Details (row, col, reg, bit): {list(zip(self.fault_pe_row, self.fault_pe_col, self.fault_reg, self.fault_bit))}")
        else:
            print(f"  Fault PE Position: {list(zip(self.fault_pe_row, self.fault_pe_col))}")
        print(f"  Enabled: {self.enabled}")

    def parse_fault_type(self):
        parts = self.fault_type_str.split('_')
        mode = parts[0]
        f_type = parts[1]
        
        op = 'flip'
        pos = 0
        if f_type == 'bitflip':
            op = 'flip'
            pos = int(parts[2]) if len(parts) > 2 else 0
        elif f_type == 'stuck':
            val = parts[2]
            op = f'stuck_{val}'
            pos = int(parts[3]) if len(parts) > 3 else 0
            
        self.fault_config = {
            'mode': mode,
            'op': op,
            'pos': pos
        }

    def set_fault_config_injpos(self, val):
        self.fault_config['pos'] = val

    def reset_fault_pe(self):
        self.fault_pe_row = []
        self.fault_pe_col = []
        self.fault_reg = []
        self.fault_bit = []
        self._pe_mask_cache = {}  # 清除缓存

    def init_fault_position(self):
        self.reset_fault_pe()
        self.fault_pe_row.append(random.randint(0, self.sa_rows - 1))
        self.fault_pe_col.append(random.randint(0, self.sa_cols - 1))
        self.fault_reg.append(0)
        self.fault_bit.append(self.fault_config.get('pos', 0))

    def init_multi_fault_positions(self, num_faults: int, num_regs: int = 1, num_bits: int = 16):
        self.reset_fault_pe()
        max_pe_positions = self.sa_rows * self.sa_cols
        max_positions = max_pe_positions * num_regs * num_bits
        if num_faults <= 0: return
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
        self.fault_bit.append(self.fault_config.get('pos', 0))

    def set_multi_fault_positions(self, positions: list):
        self.reset_fault_pe()
        for row, col in positions:
            if not (0 <= row < self.sa_rows and 0 <= col < self.sa_cols):
                raise ValueError("PE bounds exceeded")
            self.fault_pe_row.append(row)
            self.fault_pe_col.append(col)
            self.fault_reg.append(0)
            self.fault_bit.append(self.fault_config.get('pos', 0))

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
        # 缓存到实例变量：只有当故障位置集合发生变化时才重建
        # 用故障列表的长度作为简单指纹（位置通常只设置一次）
        fingerprint = len(self.fault_pe_row)
        device_key = str(device)
        cached = self._pe_mask_cache.get(device_key)
        if cached is not None and cached[1] == fingerprint:
            return cached[0]
        
        # 首次或故障集合变更时重建
        pe_mask_map_cpu = torch.zeros((self.sa_rows, self.sa_cols), dtype=torch.int32)
        op = self.fault_config['op']
        
        for f_idx in range(len(self.fault_pe_row)):
            r = self.fault_pe_row[f_idx]
            c = self.fault_pe_col[f_idx]
            pos = self.fault_bit[f_idx]
            
            mask_pos = pos + 16 if self.precision == 'bf16' and pos < 16 else pos
            bit = 1 << mask_pos
            
            if op == 'flip':
                pe_mask_map_cpu[r, c] ^= bit
            elif op == 'stuck_1':
                pe_mask_map_cpu[r, c] |= bit
            elif op == 'stuck_0':
                pe_mask_map_cpu[r, c] |= bit
                
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
        
        r_f = torch.tensor(self.fault_pe_row, device=device).view(-1, 1, 1)
        c_f = torch.tensor(self.fault_pe_col, device=device).view(-1, 1, 1)
        
        if self.dataflow == 'WS':
            Y_faulty = self._simulate_ws(X_2d, W, M, K, N, r_f, c_f, device)
        elif self.dataflow == 'OS':
            Y_faulty = self._simulate_os(X_2d, W, M, K, N, r_f, c_f, device)
        elif self.dataflow == 'IS':
            Y_faulty = self._simulate_is(X_2d, W, M, K, N, r_f, c_f, device)
        else:
            raise ValueError(f"Unknown dataflow: {self.dataflow}")
            
        Y_faulty = torch.clamp(Y_faulty, -1e3, 1e3)
        return Y_faulty.view(original_shape[:-1] + (N,))

    def _simulate_ws(self, X, W, M, K, N, r_f, c_f, device):
        # WS Mapping: W_{k, j} -> PE(k % R, j % C)
        k_mod = (torch.arange(K, device=device) % self.sa_rows).unsqueeze(1) # [K, 1]
        j_mod = (torch.arange(N, device=device) % self.sa_cols).unsqueeze(0) # [1, N]
        
        mode = self.fault_config['mode']
        op = self.fault_config['op']
        pe_mask_map = self._build_pe_mask_map(device) # [R, C]
        
        if mode == 'weight':
            W_tilde = W.clone()
            # Broadcast mask map to W shape
            mapped_mask = pe_mask_map[k_mod.expand(K, N), j_mod.expand(K, N)]
            active_mask = mapped_mask > 0 if op != 'stuck_0' else mapped_mask != 0 # Check if any operations exist
            if active_mask.any():
                W_tilde[active_mask] = self._inject_bit_error(W_tilde[active_mask], mapped_mask[active_mask], op)
            return torch.matmul(X, W_tilde)
            
        elif mode == 'input':
            Y_tilde = torch.matmul(X, W)
            # Find columns in the PE array that have faults
            active_cols = torch.nonzero(pe_mask_map.sum(dim=0)).squeeze(-1)
            if active_cols.dim() == 0: active_cols = active_cols.unsqueeze(0)
            
            for c in active_cols:
                col_mask = pe_mask_map[:, c] # [R]
                k_mask_for_c = col_mask[k_mod.squeeze(-1)] # [K]
                
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
                X_flipped[active_2d] = self._inject_bit_error(X_flipped[active_2d], mask_2d[active_2d], op)
                
                dx = X_flipped - X # [M, K]
                
                in_mask = (j_mod >= c).squeeze(0) # [N]
                W_masked = W * in_mask.to(W.dtype) # [K, N]
                Y_tilde += torch.matmul(dx, W_masked)
                
            return Y_tilde
            
        elif mode == 'psum':
            Y = torch.matmul(X, W)
            Y_tilde = Y.clone()
            
            active_rc = torch.nonzero(pe_mask_map)
            for point in active_rc:
                r, c = point[0].item(), point[1].item()
                mask_val = pe_mask_map[r, c]
                
                # Affects Y[:, j] where j % sa_cols == c
                affected_j = torch.where((torch.arange(N, device=device) % self.sa_cols) == c)[0]
                
                scale = max((K - r) / K, 0.0) if K > 0 else 1.0
                if affected_j.numel() > 0:
                    col_data = Y[:, affected_j]
                    # mask_val is a scalar int32 for this r,c 
                    mask_tensor = mask_val.expand(col_data.shape)
                    flipped = self._inject_bit_error(col_data, mask_tensor, op)
                    diff = flipped - col_data
                    Y_tilde[:, affected_j] += diff * scale
                    
            return Y_tilde
            
        return torch.matmul(X, W)

    def _simulate_os(self, X, W, M, K, N, r_f, c_f, device):
        # OS Mapping: Y_{i, j} -> PE(i % R, j % C)
        i_mod = (torch.arange(M, device=device) % self.sa_rows).unsqueeze(1) # [M, 1]
        j_mod = (torch.arange(N, device=device) % self.sa_cols).unsqueeze(0) # [1, N]
        
        Y = torch.matmul(X, W)
        mode = self.fault_config['mode']
        op = self.fault_config['op']
        pe_mask_map = self._build_pe_mask_map(device)
        
        if mode == 'psum':
            Y_tilde = Y.clone()
            mapped_mask = pe_mask_map[i_mod.expand(M, N), j_mod.expand(M, N)]
            active_mask = mapped_mask > 0 if op != 'stuck_0' else mapped_mask != 0
            if active_mask.any():
                Y_tilde[active_mask] = self._inject_bit_error(Y_tilde[active_mask], mapped_mask[active_mask], op)
            return Y_tilde
            
        elif mode == 'input':
            Y_tilde = Y.clone()
            # Loop over unique (r, c) PE positions with non-zero masks; 
            # upper bound is sa_rows * sa_cols = 65536, much smaller than raw BER count.
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
            
        elif mode == 'weight':
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
        # IS Mapping: X_{i, k} -> PE(i % R, k % C)
        i_mod = (torch.arange(M, device=device) % self.sa_rows).unsqueeze(1) # [M, 1]
        k_mod = (torch.arange(K, device=device) % self.sa_cols).unsqueeze(0) # [1, K]
        
        Y = torch.matmul(X, W)
        mode = self.fault_config['mode']
        op = self.fault_config['op']
        pe_mask_map = self._build_pe_mask_map(device)
        
        if mode == 'input':
            X_tilde = X.clone()
            mapped_mask = pe_mask_map[i_mod.expand(M, K), k_mod.expand(M, K)]
            active_mask = mapped_mask > 0 if op != 'stuck_0' else mapped_mask != 0
            if active_mask.any():
                X_tilde[active_mask] = self._inject_bit_error(X_tilde[active_mask], mapped_mask[active_mask], op)
            return torch.matmul(X_tilde, W)
            
        elif mode == 'weight':
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
            
        elif mode == 'psum':
            Y_tilde = Y.clone()
            active_rc = torch.nonzero(pe_mask_map)
            for point in active_rc:
                r, c = point[0].item(), point[1].item()
                mask_val = pe_mask_map[r, c]
                
                # Affects Y[i, :] where i % sa_rows == r
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
