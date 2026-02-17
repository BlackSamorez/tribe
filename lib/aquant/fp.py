from lib.utils.matmul_had import grouped_hadamard

import torch


FP4_LEVELS = torch.tensor([
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
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
        
    if aquant == 'bf16':
        return x
    elif aquant == 'fp8':
        if x.isnan().any():
            print(x)
            raise Exception("A")
        x = x.reshape(-1, group_size)
        x_scales = x.abs().max(dim=-1, keepdim=True).values / 447.99
        x_scales[x_scales == 0.0] = 1.0
        x /= x_scales
        x = x.to(torch.float8_e4m3fn).float()
        x *= x_scales
        
        if x.isnan().any():
            print(x)
            raise Exception("B")
        
        return x.reshape(orig_shape)
    elif aquant == 'fp4_absmax':
        x = x.reshape(-1, group_size)
        x_scales = x.abs().max(dim=-1, keepdim=True).values / FP4_LEVELS.max()
        x_scales[x_scales == 0.0] = 1.0
        x /= x_scales
        x = rtn_fp4(x)
        x *= x_scales
        return x.reshape(orig_shape)
    elif aquant == 'fp4_quest':
        x = x.reshape(-1, group_size)
        x_scales = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_scales[x_scales == 0.0] = 1.0
        x /= x_scales
        x *= (6.0 / 2.92247856) # MSE-optimal clipping
        x = rtn_fp4(x)
        x *= x_scales
        return x.reshape(orig_shape)
    elif aquant == "nvfp4":
        assert group_size == 16
        x = x.view(-1, 16)
        scales = x.abs().max(dim=-1, keepdim=True)[0]

        s_dec = scales.max() / (447.99 * 6.0)
        s_dec[s_dec == 0] = 1.0
        s_dec_b = scales / 6.0
        s_dec_b_e4m3 = (s_dec_b / s_dec).to(torch.float8_e4m3fn).float()
        s_dec_b_e4m3[s_dec_b_e4m3 == 0] = 1.0
        s_enc_b_inv = s_dec_b_e4m3 * s_dec
        
        x = rtn_fp4(
            torch.clamp(x / s_enc_b_inv, -5.99, 5.99)
        ) * s_enc_b_inv
        return x.reshape(orig_shape)
    else:
        raise ValueError(f"Invalid aquant: {aquant}")
