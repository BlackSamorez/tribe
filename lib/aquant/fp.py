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


@torch.compile
def quantize_activations(x: torch.Tensor, aquant: str, group_size: int, hadamard_size: int):
    orig_shape = x.shape
    x = grouped_hadamard(x, hadamard_size)
        
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
    elif aquant == "nvfp4":
        assert group_size == 16
        x_grouped = x.view(-1, 16)
        scales = x_grouped.abs().max(dim=-1, keepdim=True)[0]

        s_dec = scales.max() / (447.99 * 6.0)
        s_dec[s_dec == 0] = 1.0
        s_dec_b = scales / 6.0
        s_dec_b_e4m3 = (s_dec_b / s_dec).to(torch.float8_e4m3fn).float()
        s_dec_b_e4m3[s_dec_b_e4m3 == 0] = 1.0
        s_enc_b_inv = s_dec_b_e4m3 * s_dec
        x = rtn_fp4(x_grouped / s_enc_b_inv) * s_enc_b_inv
        x = x.reshape(orig_shape)
    else:
        raise ValueError(f"Invalid aquant: {aquant}")
    return x
