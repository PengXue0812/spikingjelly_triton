import torch
import triton
import triton.language as tl

@triton.jit
def get_grad_s_to_h(over_th_ptr, alpha, BLOCK_SIZE):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < BLOCK_SIZE
    over_th = tl.load(over_th_ptr, offset, mask=mask)
    sg_ATan_M_PI_2__alpha__x = (1.57079632679489661923) * alpha * over_th
    grad_s_to_h = alpha / 2. / (1. + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x)
    return grad_s_to_h



@triton.jit
def test(x_ptr, y_ptr, alpha, BLOCK_SIZE):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < BLOCK_SIZE
    x = tl.load(x_ptr, offset, mask=mask)
    y = alpha * tl.atan(x)
    tl.store(y_ptr, y)