from typing import Optional

import torch

from lib.utils.matmul_had import grouped_hadamard

@torch.compile
def scale_weight(
    Wr: torch.Tensor,
    group_size: int,
    codebook_std: torch.Tensor,
    scale_override: float,
    extra_scaling_scheme: Optional[str],
) -> [torch.Tensor, torch.Tensor]:
    if extra_scaling_scheme == "no":
        Wscale = Wr.reshape(-1, group_size).square().mean(dim=-1, keepdim=True).sqrt()
    elif extra_scaling_scheme == "nvfp4":
        assert group_size == 16        
        Wscale = Wr.reshape(-1, group_size).square().mean(dim=-1, keepdim=True).sqrt()
        Wscale_max = Wscale.max() / 447.99
        Wscale = (Wscale / Wscale_max).to(torch.float8_e4m3fn).float() * Wscale_max
    else:
        raise ValueError(f"Unknown extra_scaling_scheme: {extra_scaling_scheme}")

    Wscale = Wscale / (codebook_std * scale_override)
    return (Wr.reshape(-1, group_size) / Wscale).reshape_as(Wr), Wscale
