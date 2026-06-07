from contextlib import nullcontext

import torch
import spconv.pytorch as spconv


def cast_sparse_conv_tensor_fp32(tensor):
    if not isinstance(tensor, spconv.SparseConvTensor):
        return tensor
    if tensor.features.dtype == torch.float32:
        return tensor
    return tensor.replace_feature(tensor.features.float())


def spconv_eval_fp32(module, tensor):
    if module.training or not isinstance(tensor, spconv.SparseConvTensor):
        return module(tensor)

    tensor = cast_sparse_conv_tensor_fp32(tensor)
    if tensor.features.is_cuda:
        autocast_off = torch.amp.autocast(device_type="cuda", enabled=False)
    else:
        autocast_off = nullcontext()

    with autocast_off:
        return module(tensor)
