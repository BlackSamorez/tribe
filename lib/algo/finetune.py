"""
Utilities for fine tuning
"""
import copy
import math
from contextlib import contextmanager
from operator import attrgetter

import glog
import torch
from torch import multiprocessing as mp
from torch import nn
from transformers import AutoModelForCausalLM

from lib import codebook, utils
from lib.linear import QuantizedLinear
from lib.wscale.fp import scale_weight
from lib.wscale.wush import get_xvsh_wush
from lib.aquant.fp import apply_wush, grouped_hadamard

from . import ldlq


@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, orig_dtype,
                           args):
    if args.ft_epochs == 0:
        return

    with use_tf32():
        layer = layer.to(device)

        source = next(iter(train_dl))[0]
        position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
        # manifest tensor parallel attributes in layer
        output = layer(source.to(device),
                       position_ids=position_ids)[0]
        
        best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
        utils.clean()

        optim = torch.optim.Adam(layer.parameters(), lr=args.ft_lr)
        best_loss = utils.calculate_mse_loss(layer, valid_dl, device)
        glog.info(f'layer {name} initial loss {best_loss}')
        scaler = torch.cuda.amp.GradScaler(enabled=(orig_dtype==torch.float16))
        worse_ct = 0

        for epoch in range(args.ft_epochs):
            for bidx, (source, targets) in enumerate(train_dl):
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda',
                                    dtype=orig_dtype,
                                    enabled=True):
                    output = layer(source.to(device),
                                   position_ids=position_ids)[0]
                    loss = nn.MSELoss()(output, targets)
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                        train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                test_loss = utils.calculate_mse_loss(layer, valid_dl, device)
                if test_loss < best_loss:
                    glog.info(
                        f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                    )
                    best_loss = test_loss
                    best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
                    utils.clean()
                    worse_ct = 0
                else:
                    glog.info(
                        f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                    )
                    worse_ct += 1
                    if worse_ct >= args.ft_early_stop:
                        break

    del optim, train_dl, valid_dl

    layer = layer.cpu()
    layer.load_state_dict(best_sd)
    utils.clean()


def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args,
                                    device, pre_orig_emb, orig_emb, group_size, hadamard_size):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_grad_enabled(False)

    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    orig_dtype = None
    for p in mixed_layer.parameters():
        orig_dtype = p.dtype
        break
    mixed_layer = mixed_layer.float()

    if args.ft_epochs == 0:
        train_dl, valid_dl = None, None
    else:
        train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    has_kernel = utils.has_kernel(args.decode_mode, args.L, args.K, args.V,
                                  args.tlut_bits, args.td_x, args.td_y)

    for quant_i, (linear_attr, name, in_hess_name, out_hess_name) in enumerate(quant_order):
        utils.clean()
        cb = cb.to(device).to(orig_dtype)
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        W = orig_linear.weight.to(dtype_)
        del orig_linear
        (m, n) = W.shape

        in_hess_path = f'{args.in_hess_path}/{idx}_{in_hess_name}.pt'
        H_data = torch.load(in_hess_path, map_location=torch.device('cpu'), weights_only=False)
        HR = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        if 'mu' in H_data:
            mu = H_data['mu']
            HR += mu[None, :] * mu[:, None]
            del mu
        del H_data

        HR = utils.regularize_H(HR, args.sigma_reg)
        
        if args.wush != 'no':
            Wr = W.to(device)
            HRr = HR.to(device)
            
            xvsh, wush = get_xvsh_wush(
                Wr, HRr, hadamard_size,
                ignore_weight=(args.wush == "act"),
            )
            
            Wr = apply_wush(Wr, wush)
            HRr = apply_wush(apply_wush(HRr, xvsh).T, xvsh).T
            HRr = utils.regularize_H(HRr, args.sigma_reg)
        else:
            xvsh = None
            Wr = utils.grouped_hadamard(W.to(device), hadamard_size)
            HRr = utils.grouped_hadamard(
                utils.grouped_hadamard(
                    HR.to(device), hadamard_size
                ).T,
                hadamard_size,
            ).T.reshape_as(HR)
            

        Wr, Wscale = scale_weight(Wr, group_size=group_size, codebook_std=cb.lut.to(torch.float64).square().mean().sqrt().float(), scale_override=args.scale_override, extra_scaling_scheme=args.extra_wscaling_scheme)

        LRr, _ = utils.block_LDL(HRr, args.td_y)
        diag = torch.arange(n, device=LRr.device)
        LRr[diag, diag] = 0

        hatWr, Qidxs = ldlq.LDLQ(Wr, LRr, cb, args, for_kernel=has_kernel)

        Qidxs = Qidxs.cpu()
        packed = cb.pack_trellis(
            Qidxs.reshape(m // args.td_x, args.td_x, n // args.td_y,
                          args.td_y // args.V).transpose(1, 2).reshape(
                              -1, args.td_x * args.td_y // args.V))

        if has_kernel:
            packed = packed.view(torch.uint8).view(-1, 2).flip(
                (-1, )).reshape(m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8,
                                args.K).permute(0, 2, 4, 3, 1, 5).flip(
                                    (-1, )).contiguous().flatten().view(
                                        torch.int16).reshape(packed.shape)
        else:
            packed = packed.view(torch.int16)

        Wr = (Wr.reshape(-1, group_size) * Wscale).reshape_as(Wr)
        hatWr = (hatWr.reshape(-1, group_size) * Wscale).reshape_as(hatWr)

        err = torch.trace(
            (Wr - hatWr) @ HRr @ (Wr - hatWr).T) / torch.trace(Wr @ HRr @ Wr.T)
        # print(
        #     f'{idx}_{name} proxy err {err.item()} tr(WHW.T) {torch.trace(Wr @ HRr @ Wr.T)}'
        # )
        
        save_path = f'{args.save_path}/{idx}_{name}.pt'

        # 0 = no tensor parallelism, 1 = row parallel, 2 = column parallel

        torch.save(
            {
                'trellis':
                packed.cpu(),
                'Wscale':
                Wscale,
                'proxy_err':
                err.item(),
                'tlut':
                cb.tlut.data.to(orig_dtype).cpu()
                if hasattr(cb, 'tlut') else None,
                'xvsh': xvsh.cpu() if xvsh is not None else None, 
            }, save_path)


        del HRr, Wr, hatWr, LRr, Qidxs
        utils.clean()
        
        q_linear = QuantizedLinear(
            n,
            m,
            args.td_x,
            args.td_y,
            args.L,
            args.K,
            args.V,
            args.tlut_bits,
            args.decode_mode,
            mode='train-recons' if args.ft_train_lut else 'train-fixW',
            group_size=group_size,
            hadamard_size=hadamard_size,
            xvsh=xvsh is not None,
            aquant=args.aquant,
            dtype=orig_dtype,
            grad_ckpt=args.ft_grad_ckpt)
        q_linear.trellis.copy_(packed)
        q_linear.scales.copy_(Wscale)
        q_linear = q_linear.to(device).float()

        del packed
        utils.clean()
        
        q_linear.scales = nn.Parameter(q_linear.scales, requires_grad=True)

        if q_linear.tlut is not None:
            q_linear.tlut.copy_(cb.tlut.data)
            q_linear.tlut.requires_grad = args.ft_train_lut
        
        if q_linear.xvsh is not None:
            q_linear.xvsh.copy_(xvsh.data)

        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            q_linear)

        with torch.enable_grad():
            finetune_decoder_layer(mixed_layer, f'{idx}_{name}', device,
                                   train_dl, valid_dl, orig_dtype, args)

        cb = cb.cpu()
        utils.clean()

    for quant_i, (linear_attr, name, in_hess_name, out_hess_name) in enumerate(quant_order):
        quant_linear = attrgetter(linear_attr)(mixed_layer)
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        data = torch.load(save_path, weights_only=False)

        if quant_linear.tlut is not None:
            data['tlut'] = quant_linear.tlut.data.to(orig_dtype).cpu()
        torch.save(data, save_path)

    mixed_layer = mixed_layer.to(orig_dtype).cpu()

    utils.clean()
    torch.set_grad_enabled(False)


def infer(args, end_dev, n_layers, in_q, out_q):
    with torch.no_grad():
        fake_dev_map = {
            'model.embed_tokens': 0,
            'model.rotary_emb': 0,
            'model.norm': end_dev - 1,
            'lm_head': end_dev - 1
        }
        per_dev = math.ceil(n_layers / end_dev)
        for i in range(n_layers):
            fake_dev_map[f'model.layers.{i}'] = (i + 1) // per_dev

        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype='auto',
                                                     device_map=fake_dev_map,
                                                     low_cpu_mem_usage=True)
        while True:
            data = in_q.get()
            if data is None:
                return
            out_q.put(
                model(data.to(0))['logits'][:, :-1].contiguous().softmax(
                    dim=-1).cpu())


def finetune_susv_e2e(quant_model, start_dev, devset, orig_dtype, args):

    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=infer,
                   args=(args, start_dev, len(quant_model.model.layers), in_q,
                         out_q))
    p.start()

    train_dl, valid_dl = utils.split_data(devset, devset, args)

    optim = torch.optim.Adam(quant_model.parameters(), lr=args.ft_lr)

    best_loss = utils.calculate_ce_loss_model(quant_model, valid_dl, start_dev,
                                              in_q, out_q)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(quant_model.state_dict())
    glog.info(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, _) in enumerate(train_dl):
            in_q.put(source)
            with torch.autocast(device_type='cuda',
                                dtype=orig_dtype,
                                enabled=True):
                output = quant_model(
                    source.to(start_dev))['logits'][:, :-1].contiguous()
                target = out_q.get().to(output.device)
                target = target.view(-1, target.shape[-1])
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             target)
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_ce_loss_model(quant_model, valid_dl,
                                                      start_dev, in_q, out_q)
            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(quant_model.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    in_q.put(None)
    p.join()
    with torch.no_grad():
        quant_model.load_state_dict(best_sd)
