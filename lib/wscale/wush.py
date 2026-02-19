import torch

from scipy.linalg import hadamard


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device, **kwargs):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )


def get_xvsh_wush(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    hadamard_size: int,
    ignore_weight: bool=False,
    eps: float=1e-8,
) -> [torch.Tensor, torch.Tensor]:
    weight = weight.T
    (in_dim, out_dim) = weight.shape
    assert hessian.shape == (in_dim, in_dim)

    hessian_grouped = torch.diagonal(
        hessian.reshape(in_dim // hadamard_size, hadamard_size, in_dim // hadamard_size, hadamard_size).permute(0, 2, 1, 3),
        dim1=0, dim2=1,
    ).permute(2, 0, 1).contiguous()
    weight_grouped = weight.reshape(in_dim // hadamard_size, hadamard_size, out_dim).contiguous()

    x_prime = torch.linalg.cholesky(hessian_grouped)
    
    if ignore_weight:
        w_prime = torch.eye(weight_grouped.shape[1], device=weight_grouped.device, dtype=weight_grouped.dtype)[None,:].repeat(in_dim // hadamard_size, 1, 1) * (weight_grouped.shape[2])**-0.5
    else:
        w_prime = torch.linalg.cholesky(
            torch.bmm(weight_grouped, weight_grouped.permute(0, 2, 1)) / weight_grouped.shape[2]
            + torch.eye(weight_grouped.shape[1], device=weight_grouped.device, dtype=weight_grouped.dtype)[None,:] * eps,
        )

    U, S, Vt = torch.linalg.svd(
        torch.bmm(w_prime.permute(0, 2, 1), x_prime)
    )
    S = torch.diag_embed(S)
    S_m12 = torch.linalg.inv(S).sqrt()

    T_xvs = torch.bmm(
        torch.bmm(
            S_m12,
            Vt,
        ),
        x_prime.permute(0, 2, 1),
    )

    T_wus = torch.bmm(
        torch.bmm(
            S_m12,
            U.permute(0, 2, 1),
        ),
        w_prime.permute(0, 2, 1),
    )
    
    h = get_hadamard_matrix(hadamard_size, weight.dtype, weight.device)[None, ...].repeat(in_dim // hadamard_size, 1, 1)
    
    T_xvsh = torch.bmm(h, T_xvs)
    T_wush = torch.bmm(h, T_wus)
    
    assert T_xvsh.shape == T_wush.shape
    assert T_wush.shape == (in_dim // hadamard_size, hadamard_size, hadamard_size)
    
    return T_xvsh, T_wush
