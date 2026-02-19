import math
import time

from lib.utils.matmul_had import hadamard
import torch
import torch.nn as nn

from lib.codebook import bitshift
from lib.utils import (clean, has_kernel)
from lib.aquant.fp import quantize_activations


class QuantizedLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        td_x,
        td_y,
        L,  # trellis window
        K,  # bpw
        V,  # vq dim
        tlut_bits,  # tunable LUT bits
        decode_mode,
        bias=False,
        group_size=128,
        hadamard_size=None,
        xvsh=False,
        aquant=None,
        dtype=torch.float16,
        mode='eval',
        grad_ckpt=False,
    ):
        super().__init__()
        
        if hadamard_size is None:
            hadamard_size = group_size

        self.in_features = in_features
        self.out_features = out_features
        self.td_x = td_x
        self.td_y = td_y
        self.L = L
        self.K = K
        self.V = V
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode
        self.group_size = group_size
        self.hadamard_size = hadamard_size
        self.aquant = aquant
        self.dtype = dtype
        # packed into int16
        self.register_buffer(
            'trellis',
            torch.zeros((out_features // td_x) * (in_features // td_y),
                        math.ceil((td_x * td_y) * K / 16),
                        dtype=torch.int16))

        if decode_mode in ['lut', 'quantlut', 'quantlut_sym']:
            self.tlut = nn.Parameter(torch.zeros(2**tlut_bits,
                                                 V,
                                                 dtype=torch.float16),
                                     requires_grad=False)
        else:
            self.tlut = None
        
        if xvsh:
            self.xvsh = nn.Parameter(torch.empty(
                (in_features//hadamard_size, hadamard_size, hadamard_size),
                dtype=dtype,
                requires_grad=False,
            ))
        else:
            self.xvsh = None

        if bias:
            self.register_buffer('bias', torch.ones(out_features))
        else:
            self.bias = None

        self.register_buffer("scales", torch.ones((out_features * in_features) // group_size, 1, dtype=self.dtype))

        self.built_codebook_class = False
        self.built_graph = False
        self.mode = mode
        self.grad_ckpt = grad_ckpt
        self.has_kernel = has_kernel(decode_mode, L, K, V, tlut_bits, td_x,
                                     td_y)

    def forward(self, input):
        if self.grad_ckpt:
            return self.ckpt_forward(input)
        return self.no_ckpt_forward(input)

    def ckpt_forward(self, input):
        return torch.utils.checkpoint.checkpoint(self.no_ckpt_forward,
                                                 input,
                                                 use_reentrant=True)

    def no_ckpt_forward(self, input):
        if not self.built_codebook_class:
            self.codebook_class = bitshift.BitshiftLinear(
                self.td_x,
                self.td_y,
                self.L,
                self.K,
                self.V,
                self.tlut_bits,
                self.decode_mode,
                self.group_size,
                dtype=self.dtype,
                tlut=self.tlut,
                has_kernel=self.has_kernel)

            if self.mode == 'eval':
                pass
            elif self.mode == 'train-recons':
                if not self.has_kernel:
                    self.packed_trellis = self.trellis.cpu()
                    unpacked_trellis = self.codebook_class.cb.unpack_trellis(
                        self.trellis, self.td_x * self.td_y)
                    self.trellis = unpacked_trellis
                    clean()
            elif self.mode == 'train-fixW':
                self.codebook_class.cache_hatW(self.trellis, self.scales, self.out_features, self.in_features)
                self.trellis = self.trellis.cpu()
                clean()
            else:
                raise Exception

            self.built_codebook_class = True

        # print(
        #     "quantized_linear.py:116\n"
        #     f"\tinput: {input.shape}\n"
        #     f"\tself.trellis: {self.trellis.shape}\n"
        #     f"\tself.scales: {self.scales.shape}\n"
        #     f"\tself.out_features: {self.out_features}\n"
        #     f"\tself.in_features: {self.in_features}\n"
        #     f"\tself.mode: {self.mode}\n"
        # )
        
        input = quantize_activations(
            input,
            self.aquant,
            self.group_size,
            self.hadamard_size if self.xvsh is None else self.xvsh,
        )
        
        result = self.codebook_class(input,
                                     self.trellis,
                                     self.scales,
                                     self.out_features,
                                     self.in_features,
                                     mode=self.mode) + 0
        if self.bias is not None:
            return result + self.bias
        return result
