import itertools
import math
import os
from functools import cache

import numpy as np
import torch
from torch import nn

from lib.codebook import kdict
from lib.utils.kernel_check import has_kernel
from lib.utils.kernel_decompress import decode_compressed
from lib.aquant.fp import quantize_activations


def decode_1mad(x):
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * 34038481 + 76625530
    x = x & ((1 << 32) - 1)
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    y = y - 510
    y = y.to(torch.float32)
    y = y / 147.800537109375
    return y


def decode_2mad(x):
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * 264435761 + 1013904223
    x = x & ((1 << 32) - 1)
    x = ((x * 1664525) >> 32) + x
    x = x & ((1 << 32) - 1)
    y = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    y = y - 510
    y = y.to(torch.float32)
    y = y / 147.800537109375
    return y


def decode_3inst(x):

    def bfe16_to_fp16(x):
        x[torch.where(x >= 2**15)] -= 2**16
        return torch.tensor(x.to(torch.int16).numpy().view(np.float16))

    a = 89226354
    b = 64248484
    fpmask = 996162400
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * a + b
    mask = (1 << 15) + ((1 << 12) - 1)
    mask = (mask << 16) + mask
    res = (mask & x) ^ fpmask
    top = bfe16_to_fp16(res >> 16)
    bottom = bfe16_to_fp16(res & ((1 << 16) - 1))
    return (top + bottom).float()


def decode_3inst_fp8(x):
    return decode_3inst(x).to(torch.float8_e4m3fn).float()


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


def decode_3inst_fp4(x):
    return rtn_fp4(decode_3inst(x))


def quantlut(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L)
        lut = (lut + 1) * lut
        lut = (lut >> (16 - nbits)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    return lut


def quantlut_sym(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp = 1 - ((lut >> 15) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    lut[:, 0] = lut[:, 0] * sflp
    return lut


class bitshift_codebook(nn.Module):

    def __init__(self,
                 L=16,
                 K=2,
                 V=2,
                 tlut_bits=16,
                 decode_mode='lut',
                 tlut=None):
        super(bitshift_codebook, self).__init__()
        self.idx_dtype = torch.int32
        self.opt_scale = 1

        self.L = L
        self.K = K
        self.V = V
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode

        if decode_mode == 'lut':
            if tlut is None:
                assert tlut_bits == L
                self.register_buffer('tlut', torch.randn(2**L, V))
                self.register_buffer('lut', self.tlut.T.contiguous())
            else:
                self.tlut = tlut
                self.recons_lut()

        elif decode_mode == '1mad':
            assert V == 1
            self.register_buffer('lut',
                                 decode_1mad(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == '2mad':
            assert V == 1
            self.register_buffer('lut',
                                 decode_2mad(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == '3inst':
            assert V == 1
            self.register_buffer('lut',
                                 decode_3inst(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == '3inst_fp8':
            assert V == 1
            self.register_buffer('lut',
                                 decode_3inst_fp8(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == '3inst_fp4':
            assert V == 1
            self.register_buffer('lut',
                                 decode_3inst_fp4(torch.arange(2**L)).unsqueeze(0))
        elif decode_mode == 'quantlut':
            if tlut is None:
                assert tlut_bits > 0
                if V == 1:
                    tlut = torch.erfinv((torch.arange(1 << tlut_bits) + 0.5) /
                                        (1 << tlut_bits) * 2 -
                                        1) * torch.tensor(2.0).sqrt()
                elif V == 2:
                    n = 2**tlut_bits
                    tlut = torch.zeros(n)
                    R = ((n / (n - torch.arange(n))).log() * 2).sqrt()
                    tlut = torch.stack(
                        [R * torch.arange(n).sin(), R * torch.arange(n).cos()],
                        dim=-1)
                else:
                    raise Exception
                self.register_buffer('tlut', tlut.unsqueeze(-1))
                self.register_buffer(
                    'lut',
                    quantlut(self.tlut, L, tlut_bits).T.contiguous())
            else:
                self.tlut = tlut
                self.recons_lut()
        elif decode_mode == 'quantlut_sym':
            if tlut is None:
                assert tlut_bits > 0
                if V == 2:
                    fname = f'/tmp/kmeans_{tlut_bits}_{V}.pt'
                    if not os.path.exists(fname):
                        tlut = torch.randn(2**tlut_bits, V)
                        import scipy
                        data = torch.randn(1 << 20, 2)
                        clusters = scipy.cluster.vq.kmeans(data, tlut)
                        tlut = torch.tensor(clusters[0])
                        tlut = (tlut /
                                tlut.std(unbiased=False)) * 0.9682458365518543
                        torch.save(tlut, fname)
                    else:
                        tlut = torch.load(fname)
                else:
                    raise Exception
                self.register_buffer('tlut', tlut)
                self.register_buffer(
                    'lut',
                    quantlut_sym(self.tlut, L, tlut_bits).T.contiguous())
            else:
                self.tlut = tlut
                self.recons_lut()
        else:
            raise Exception

        self.fakeinf = torch.tensor(torch.inf)

        self.register_buffer('sumdelta',
                             torch.arange(2**(K * V)) << (L - K * V))
        self.sumdelta = self.sumdelta.view(1, 1, -1)

        self.register_buffer('state', torch.arange(2**L).unsqueeze(0))
        self.register_buffer('state_cand',
                             (self.state >>
                              (K * V))[0, ::2**(K * V)].unsqueeze(-1) +
                             self.sumdelta)
        self.register_buffer('recons_state', self.recons(self.state))

        self.version = 0

    def recons_lut(self):
        if self.decode_mode == 'lut':
            self.lut = self.tlut.T.contiguous()
        elif self.decode_mode == 'quantlut':
            self.lut = quantlut(self.tlut, self.L,
                                self.tlut_bits).T.contiguous()
        elif self.decode_mode == 'quantlut_sym':
            self.lut = quantlut_sym(self.tlut, self.L,
                                    self.tlut_bits).T.contiguous()

    def recons(self, encoded, **kwargs):
        return self.lut[:,
                        encoded.int().to(self.lut.device)].to(encoded.device)

    @torch.compile
    def update(self, cost, thing):
        state_err = (self.recons_state -
                     thing.unsqueeze(-1)).square().sum(dim=0)
        cand_cost = torch.gather(
            cost.unsqueeze(-2).expand(-1, self.state_cand.shape[1], -1), -1,
            self.state_cand.expand(len(cost), -1, 2**(self.K * self.V)))
        best = torch.min(cand_cost, dim=-1)
        cost = state_err + best.values.unsqueeze(-1).expand(
            -1, -1, 2**(self.K * self.V)).reshape(state_err.shape)
        prev_state = torch.gather(
            self.state_cand.expand(thing.shape[1], -1, -1), -1,
            best.indices.unsqueeze(-1))[..., 0]
        return prev_state, cost

    def viterbi(self, X, overlap=None):
        T, B = X.shape
        assert T % self.V == 0
        # cost is (B, 2**L)
        cost = (self.recons_state -
                X[:self.V].unsqueeze(-1)).square().sum(dim=0)

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * self.fakeinf
            allow = (overlap <<
                     (self.K * self.V)).unsqueeze(-1) + torch.arange(
                         2**(self.K * self.V)).to(X.device).view(1, 1, -1)
            mask.scatter_(1, allow[0], 0)
            cost = torch.min(cost + mask, self.fakeinf)

        from_state = torch.zeros(T // self.V,
                                 B,
                                 2**(self.L - self.K * self.V),
                                 dtype=self.state.dtype,
                                 device=self.state.device)

        for i in range(1, T // self.V):
            from_state[i], cost = self.update(cost,
                                              X[i * self.V:(i + 1) * self.V])

        if overlap is not None:
            mask = torch.ones(B, 2**self.L, device=X.device) * self.fakeinf
            allow = (overlap.unsqueeze(-1) + self.sumdelta.unsqueeze(0))
            mask.scatter_(1, allow[0, 0], 0)
            cost = torch.min(cost + mask, self.fakeinf)

        final_state = torch.zeros(T // self.V,
                                  B,
                                  dtype=self.idx_dtype,
                                  device=X.device)
        final_state[T // self.V - 1] = torch.argmin(cost, dim=-1)
        for i in range(T // self.V - 1, 0, -1):
            final_state[i - 1] = torch.gather(
                from_state[i], -1,
                (final_state[i].to(torch.int64).unsqueeze(-1)) >>
                (self.K * self.V))[..., 0]
        return final_state

    def quantize_seq(self, X, overlap=None, **kwargs):
        T, NO = X.shape
        bs = min(2**(24 - self.L), NO)
        pad_amt = math.ceil(NO / bs) * bs - NO
        X = torch.nn.functional.pad(X, (0, pad_amt))
        T, N = X.shape
        X = X.reshape(T, N // bs, bs).transpose(0, 1).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap, (0, pad_amt))
            overlap = overlap.reshape(N // bs, bs)

        Qidxs = torch.zeros(N // bs,
                            T // self.V,
                            bs,
                            dtype=self.idx_dtype,
                            device=X.device)
        for i in range(len(X)):
            b_overlap = None if overlap is None else overlap[i]
            Qidxs[i] = self.viterbi(X[i], overlap=b_overlap)
        Qidxs = Qidxs.transpose(0, 1).reshape(T // self.V, N)[:, :NO]
        return Qidxs

    def quantize(self, X, **kwargs):
        X = X.T.contiguous().to(torch.float16)
        T = X.shape[0]
        roll_X = torch.roll(X, T // (2 * self.V) * self.V, 0)
        state = self.quantize_seq(roll_X, overlap=None)
        overlap = state[T // (2 * self.V)] >> self.K * self.V
        state = self.quantize_seq(X, overlap=overlap)
        hatX = self.recons(state).transpose(0, 1).reshape(X.shape)
        return hatX.T.contiguous().to(X.device), state.T.contiguous().to(
            X.device)

    def pack_trellis(self, trellis):
        # T is really T // self.V here
        B, T = trellis.shape
        bf = torch.zeros(B,
                         T * self.K * self.V + self.L - self.K * self.V,
                         dtype=bool,
                         device=trellis.device)
        bf[:, :self.L] = (trellis[:, 0].unsqueeze(-1) & (2**torch.arange(
            self.L, device=trellis.device).flip(dims=(-1, ))).unsqueeze(0)) > 0
        K_mask = 2**torch.arange(
            self.K * self.V,
            device=trellis.device).flip(dims=(-1, )).unsqueeze(0)
        for i in range(1, T):
            assert ((trellis[:, i - 1] &
                     ((1 << (self.L - self.K * self.V)) - 1)) == (
                         trellis[:, i] >> (self.K * self.V))).all()
            bf[:,
               (self.L +
                (i - 1) * self.K * self.V):(self.L + i * self.K * self.V)] = (
                    (trellis[:, i] &
                     ((1 <<
                       (self.K * self.V)) - 1)).unsqueeze(-1) & K_mask) > 0

        bf = bf[:, :-(self.L - self.K * self.V)]
        pad_amt = math.ceil(
            T * self.K * self.V / 16) * 16 - T * self.K * self.V
        bf = torch.nn.functional.pad(bf, (0, pad_amt)).reshape(
            -1, (T * self.K * self.V + pad_amt) // 16, 16)

        uint_mask = (2**torch.arange(
            16, dtype=torch.int32,
            device=bf.device)).flip(dims=(-1, )).unsqueeze(0).unsqueeze(0)
        bf_sum = (bf.to(torch.int32) * uint_mask).sum(dim=-1)
        return bf_sum.to(torch.uint16)

    def unpack_trellis(self, packed, T):
        packed = packed.view(torch.uint16).to(torch.int32)
        uint_mask = (2**torch.arange(
            16, dtype=torch.int32,
            device=packed.device)).flip(dims=(-1, )).unsqueeze(0).unsqueeze(0)
        bf = (packed.unsqueeze(-1) & uint_mask) > 0
        pad_amt = math.ceil(T * self.K / 16) * 16 - T * self.K
        bf = bf.reshape(-1, (T * self.K + pad_amt))[:, :T * self.K]
        bf = torch.concat([bf, bf[:, :self.L - self.K * self.V]], dim=-1)
        L_mask = (2**torch.arange(
            self.L, dtype=torch.int32,
            device=packed.device).flip(dims=(-1, ))).unsqueeze(0)
        K_mask = (2**torch.arange(
            self.K * self.V, dtype=torch.int32,
            device=packed.device).flip(dims=(-1, ))).unsqueeze(0)
        trellis = torch.zeros(bf.shape[0],
                              T // self.V,
                              dtype=torch.int32,
                              device=bf.device)
        trellis[:, 0] = (bf[:, :self.L].int() * L_mask).sum(dim=-1)
        for i in range(1, T // self.V):
            trellis[:, i] = ((trellis[:, i-1] << (self.K*self.V)) & ((1 << self.L) - 1)) + \
                (bf[:, self.L + (i-1)*self.K*self.V : self.L + i*self.K*self.V].int() * K_mask).sum(dim=-1)

        return trellis


class BitshiftLinear(nn.Module):

    def __init__(self,
                 td_x,
                 td_y,
                 L,
                 K,
                 V,
                 tlut_bits,
                 decode_mode,
                 group_size,
                 skip_hadamard,
                 aquant=None,
                 dtype=torch.float16,
                 tlut=None,
                 has_kernel=False):
        super().__init__()
        self.td_x = td_x
        self.td_y = td_y
        self.V = V
        self.cb = bitshift_codebook(L, K, V, tlut_bits, decode_mode, tlut=tlut)
        self.group_size = group_size
        self.skip_hadamard = skip_hadamard
        self.aquant = aquant
        self.internal_dtype = dtype
        self.has_kernel = False # TODO: add kernel

    def get_hatW(self, unpacked_trellis, m, n):
        # print(
        #     f"unpacked_trellis: {unpacked_trellis.shape}\n"
        #     f"\tm: {m}\n"
        #     f"\tn: {n}"
        # )
        # raise Exception("Stop here")
        return self.cb.recons(unpacked_trellis).transpose(0, 1).transpose(
            1, 2).reshape(m // self.td_x, n // self.td_y, self.td_x,
                          self.td_y).transpose(1, 2).reshape(m, n)

    def get_hatW_kernel(self, trellis, m, n):
        out = decode_compressed(self.cb.L, self.cb.tlut_bits, self.cb.K,
                                int(math.log2(self.V)), m, n, trellis.view(-1),
                                self.cb.lut.T)
        return out

    def cache_hatW(self, packed_trellis, scales, m, n):
        if self.has_kernel:
            hatW = self.get_hatW_kernel(packed_trellis, m, n)
        else:
            hatW = self.get_hatW(
                self.cb.unpack_trellis(packed_trellis, self.td_x * self.td_y),
                m, n)
        hatW = (hatW.reshape(-1, self.group_size) * scales).reshape_as(hatW)
        self.hatW = hatW.to(self.internal_dtype)

    def forward(self,
                input,
                trellis,
                scales,
                m, n,
                mode='eval',
                **kwargs):
        x = input.view(-1, n).to(torch.float32)

        bs = x.shape[0]

        x = quantize_activations(x, self.aquant, self.group_size, self.skip_hadamard)

        if mode == 'train-fixW':
            x = x.to(self.internal_dtype) @ self.hatW.T
            return x.view(*input.shape[:-1], m).to(input.dtype)

        if bs == 1 and self.has_kernel:
            wrapper = getattr(
                torch.ops.quip_lib,
                f"decompress_matvec_qtip_{m}_1_{x.numel()}_{self.cb.K}")

            x = wrapper(trellis, x, self.cb.tlut)
            return x.view(*input.shape[:-1], m).to(input.dtype)
        
        if mode == 'train-recons':
            self.cb.recons_lut()

        if self.has_kernel:
            x = BitshiftLinearKernelAG.apply(
                x, trellis, m, n, self.cb.L, self.cb.tlut_bits, self.cb.K,
                self.V, self.cb.lut).float()
            return x.view(*input.shape[:-1], m).to(input.dtype)
        
        if mode == 'eval':
            trellis = self.cb.unpack_trellis(
                trellis, self.td_x * self.td_y)
        hatW = self.get_hatW(trellis, m, n)
        hatW = (hatW.reshape(-1, self.group_size) * scales).reshape_as(hatW).to(self.internal_dtype)
        x = x.to(self.internal_dtype) @ hatW.T
        return x.view(*input.shape[:-1], m).to(input.dtype)

        
class BitshiftLinearKernelAG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, trellis, m, n, L, tlut_bits, K, V, lut):
        ctx.save_for_backward(trellis, lut)
        ctx.L = L
        ctx.tlut_bits = tlut_bits
        ctx.K = K
        ctx.V = V
        ctx.m = m
        ctx.n = n
        
        hatW = decode_compressed(L, tlut_bits, K, int(math.log2(V)),
                                 m, n, trellis.view(-1), lut.T)
        return input.to(hatW.dtype) @ hatW.T

    @staticmethod
    def backward(ctx, grad_output):
        trellis, lut = ctx.saved_tensors
        L = ctx.L
        tlut_bits = ctx.tlut_bits
        K = ctx.K
        V = ctx.V
        m = ctx.m
        n = ctx.n

        hatW = decode_compressed(L, tlut_bits, K, int(math.log2(V)),
                                 m, n, trellis.view(-1), lut.T)
        
        grad_input = grad_output.to(hatW.dtype) @ hatW
        return grad_input, None, None, None, None, None, None, None, None
