from lib.utils.matmul_had import grouped_hadamard

import torch


FP4_LEVELS = torch.tensor([
    -2.92247856,
    -1.94831904,
    -1.46123928,
    -0.97415952,
    -0.73061964,
    -0.48707976,
    -0.24353988,
    0.0,
    0.0,
    0.24353988,
    0.48707976,
    0.73061964,
    0.97415952,
    1.46123928,
    1.94831904,
    2.92247856,
], device="cuda")


def rtn_fp4(x):
    grid = FP4_LEVELS.to(x.device)
    inds = torch.bucketize(x, grid)

    lo = torch.clamp(inds - 1, min=0, max=15)
    hi = torch.clamp(inds,     min=0, max=15)

    g_lo = grid[lo]
    g_hi = grid[hi]

    pick_hi = (g_hi - x) <= (x - g_lo)
    return torch.where(pick_hi, g_hi, g_lo)


def quantize_activations(x: torch.Tensor, aquant: str, group_size: int, skip_hadamard: bool):
    orig_shape = x.shape
    x = grouped_hadamard(x, group_size, skip_hadamard)
        
    if aquant is None:
        pass
    elif aquant == 'fp8':
        x = x.reshape(-1, group_size)
        x_scales = x.abs().max(dim=-1, keepdim=True).values
        x /= x_scales
        x = x.to(torch.float8_e4m3fn).float()
        x *= x_scales
        x = x.reshape(orig_shape)
    elif aquant == 'fp4_absmax':
        x = x.reshape(-1, group_size)
        x_scales = x.abs().max(dim=-1, keepdim=True).values / FP4_LEVELS.max()
        x /= x_scales
        x = rtn_fp4(x)
        x *= x_scales
        x = x.reshape(orig_shape)
    elif aquant == 'fp4_quest':
        x = x.reshape(-1, group_size)
        x_scales = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x /= x_scales
        x = rtn_fp4(x)
        x *= x_scales
        x = x.reshape(orig_shape)
    else:
        raise ValueError(f"Invalid aquant: {aquant}")

    return x
