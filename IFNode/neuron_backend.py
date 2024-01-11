from typing import Any
import torch
import triton
import triton.language as tl
import spikingjelly.activation_based.surrogate as surrogate

def if_requries_grad(items):
    for item in items:
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                return True
    return False
                          
def ctx_save(ctx, requires_grad: bool, *args, **kwargs):
    if requires_grad:
        ctx.save_for_backward(*args)
        for key, value in kwargs.items():
            ctx.__setattr__(key, value)

@triton.jit
def atan_grad_s_to_h(over_th, alpha, dtype):
    sg_ATan_M_PI_2__alpha__x = ((1.57079632679489661923) * alpha * over_th).to(dtype)
    grad_s_to_h = (alpha / 2. / (1. + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x)).to(dtype)
    return grad_s_to_h

@triton.jit
def fakeNumericalGradient_grad_s_to_h(over_th, alpha):
    sg_FakeNumericalGradient_sign = (over_th >= 0.) * 2. - 1.
    grad_s_to_h = tl.math.min(sg_FakeNumericalGradient_sign / over_th, alpha)
    return grad_s_to_h

@triton.jit
def leakyKReLU_grad_s_to_h(spike, leak, k):
    sg_LeakyKReLU_mask1 = spike
    grad_s_to_h = leak * (1. - sg_LeakyKReLU_mask1) + k * sg_LeakyKReLU_mask1
    return grad_s_to_h

@triton.jit
def piecewiseLeakyReLU_grad_s_to_h(over_th, spike, w, c):
    sg_PiecewiseLeakyReLU_x_abs = tl.abs(over_th).to(spike.dtype)
    grad_s_to_h = tl.where(sg_PiecewiseLeakyReLU_x_abs > w, c, 1. / w).to(spike.dtype)
    return grad_s_to_h

@triton.jit
def qPseudoSpike_grad_s_to_h(over_th, spike, alpha):
    sg_QPseudoSpike_base = (1. + 2. / (alpha - 1.) * tl.abs(over_th)).to(spike.dtype)
    grad_s_to_h = tl.math.pow(sg_QPseudoSpike_base, -alpha).to(spike.dtype)
    return grad_s_to_h

@triton.jit
def sigmoid_grad_s_to_h(over_th, alpha):
    sigmoid_backward__sigmoid_ax = 1. / (1. + tl.exp(-alpha * over_th))
    grad_s_to_h = (1. - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * alpha
    return grad_s_to_h

@triton.jit
def s2NN_grad_s_to_h(over_th, spike, alpha, beta):
    sg_S2NN_sigmoid_ax = (1. / (1. + tl.exp(-alpha * over_th))).to(spike.dtype)
    sg_S2NN_mask_l = (over_th < 0.).to(spike.dtype)
    grad_s_to_h =  (1. - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * alpha * sg_S2NN_mask_l + beta / (over_th + 1.00005) * (1. - sg_S2NN_mask_l) 
    return grad_s_to_h

class SingleStepATGF(torch.autograd.Function):
    @staticmethod
    def new_tensors(news: tuple, py_dict: dict, ref: str='x'):
        ref = py_dict[ref]
        zero_shape = list(ref.shape)
        zero_shape[0] *= news.__len__()
        for i, item in enumerate(torch.split(torch.zeros(zero_shape, device=ref.device, dtype=ref.dtype), ref.shape[0])):
            py_dict[news[i]] = item

    @staticmethod
    def pre_forward(py_dict):
        requires_grad = if_requries_grad(py_dict.values())

        SingleStepATGF.new_tensors(('h', 'spike', 'v_next'), py_dict)
        N = py_dict['x'].numel()

        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )

        py_dict['N'] = N
        return requires_grad, grid, py_dict
    
    @staticmethod
    def pre_backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
        backward = ctx.backward
        grid = ctx.grid

        N = ctx.N
        v_th = ctx.v_th
        v_reset = ctx.v_reset
        detach_reset = ctx.detach_reset
        surrogate_function = ctx.surrogate_function    

        zero_shape = list(grad_spike.shape)
        zero_shape[0] *= 2
        zero_data = torch.zeros(zero_shape, device=grad_spike.device, dtype=grad_spike.dtype)

        real_numel = grad_spike.size(0) 
        grad_x = zero_data[0: real_numel]
        grad_v = zero_data[real_numel:]

        py_dict = {
            'grad_spike': grad_spike,
            'grad_v': grad_v,
            'grad_x': grad_x,
            'grad_v_next': grad_v_next,
            'h': ctx.h,
            'v_th': v_th,
            'v_reset': v_reset,
            'detach_reset': detach_reset,
            'N': N,
            'surrogate_function': surrogate_function,
        }
        return backward, grid, py_dict

    @staticmethod
    def forward(ctx, x: torch.Tensor, v: torch.Tensor, v_th: float, v_reset: float or None, detach_reset: bool,
                surrogate_function,forward, backward):
        py_dict = {
            'x': x,
            'v': v,
            'v_th': v_th,
            'v_reset': v_reset,
        }
        requires_grad, grid, py_dict = SingleStepATGF.pre_forward(py_dict)

        forward[grid](
            py_dict['x'],
            py_dict['v'],
            py_dict['h'],
            py_dict['spike'],
            py_dict['v_next'],
            py_dict['v_th'],
            py_dict['v_reset'],
            py_dict['N'],
        )
        ctx_save(ctx, requires_grad, h=py_dict['h'], grid=grid,
                            N=py_dict['N'],
                            v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                            backward=backward,
                            detach_reset=detach_reset,
                            surrogate_function=surrogate_function,
                            )
        
        return py_dict['spike'], py_dict['v_next']
    
    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
        backward, grid, py_dict = SingleStepATGF.pre_backward(ctx, grad_spike, grad_v_next)

        map = {'ATan':1, 'FakeNumericalGradient':2, 'LeakyKReLU':3, 'PiecewiseLeakyReLU':4, 'QPseudoSpike':5, 'Sigmoid':6, 'S2NN':7}

        surrogate_function = py_dict['surrogate_function']

        alpha = surrogate_function.alpha if hasattr(surrogate_function,'alpha') else None
        beta = surrogate_function.beta if hasattr(surrogate_function, 'beta') else None
        leak = surrogate_function.leak if hasattr(surrogate_function, 'leak') else None
        k = surrogate_function.k if hasattr(surrogate_function, 'k') else None
        w = surrogate_function.w if hasattr(surrogate_function, 'w') else None
        c = surrogate_function.c if hasattr(surrogate_function, 'c') else None

        backward[grid](
                grad_spike.contiguous(),
                py_dict['grad_v'],
                py_dict['h'],
                py_dict['grad_x'],
                py_dict['grad_v_next'],
                py_dict['v_th'],
                py_dict['v_reset'],
                py_dict['detach_reset'],
                py_dict['N'],
                surrogate_function=map[surrogate_function.__class__.__name__],
                alpha=alpha,
                beta=beta,
                leak=leak,
                k=k,
                w=w,
                c=c,
            )
        return py_dict['grad_x'], py_dict['grad_v_next'], None, None, None, None, None, None, None

class MultiStepATGF(torch.autograd.Function): 
    @staticmethod
    def new_tensors(news: tuple, py_dict: dict, ref: str='x_seq'):
        ref = py_dict[ref]
        zero_shape = list(ref.shape)
        zero_shape[0] *= news.__len__()
        for i, item in enumerate(torch.split(torch.zeros(zero_shape, device=ref.device, dtype=ref.dtype), ref.shape[0])):
            py_dict[news[i]] = item
  
    @staticmethod
    def pre_forward(py_dict):
        requires_grad = if_requries_grad(py_dict.values())
        
        MultiStepATGF.new_tensors(('h_seq', 'spike_seq', 'v_seq'), py_dict)
        py_dict['v_v_seq'] = torch.cat((py_dict.pop('v_init').unsqueeze(0), py_dict.pop('v_seq')))
        numel = py_dict['x_seq'].numel()
        N = py_dict['x_seq'].shape[1]

        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )

        py_dict['numel'] = numel
        py_dict['N'] = N
        py_dict['T'] = py_dict['x_seq'].shape[0]

        return requires_grad, grid, py_dict
    
    @staticmethod
    def pre_backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward = ctx.backward
        grid = ctx.grid

        numel = ctx.numel
        h_seq = ctx.h_seq
        N = ctx.N
        T = ctx.T
        v_th = ctx.v_th
        v_reset = ctx.v_reset
        detach_reset = ctx.detach_reset
        surrogate_function = ctx.surrogate_function

        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -1]
        grad_v_init = zero_data[-1]

        py_dict = {
            'grad_spike_seq': grad_spike_seq,
            'grad_v_seq': grad_v_seq,
            'grad_x_seq': grad_x_seq,
            'grad_v_init': grad_v_init,
            'h_seq': h_seq,
            'v_th': v_th,
            'v_reset': v_reset,
            'detach_reset': detach_reset,
            'numel': numel,
            'N': N,
            'T': T,
            'surrogate_function': surrogate_function,
        }
        return backward, grid, py_dict

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None, detach_reset: bool,
                surrogate_function, forward, backward):
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset,
        }
        requires_grad, grid, py_dict = MultiStepATGF.pre_forward(py_dict)
        forward[grid](
            x_seq,
            py_dict['v_v_seq'],
            py_dict['h_seq'],
            py_dict['spike_seq'],
            py_dict['v_th'],
            py_dict['v_reset'],
            py_dict['numel'],
            py_dict['N'],
            py_dict['T'],
        )
        ctx_save(ctx, requires_grad, h_seq=py_dict['h_seq'], grid=grid, 
                            numel=py_dict['numel'], detach_reset=detach_reset,
                            N=py_dict['N'], T=py_dict['T'],
                            v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                            backward=backward,
                            surrogate_function=surrogate_function)
        return py_dict['spike_seq'], py_dict['v_v_seq'][1:,]
        
    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward, grid, py_dict = MultiStepATGF.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        
        map = {'ATan':1, 'FakeNumericalGradient':2, 'LeakyKReLU':3, 'PiecewiseLeakyReLU':4, 'QPseudoSpike':5, 'Sigmoid':6, 'S2NN':7}

        surrogate_function = py_dict['surrogate_function']

        alpha = surrogate_function.alpha if hasattr(surrogate_function,'alpha') else None
        beta = surrogate_function.beta if hasattr(surrogate_function, 'beta') else None
        leak = surrogate_function.leak if hasattr(surrogate_function, 'leak') else None
        k = surrogate_function.k if hasattr(surrogate_function, 'k') else None
        w = surrogate_function.w if hasattr(surrogate_function, 'w') else None
        c = surrogate_function.c if hasattr(surrogate_function, 'c') else None

        backward[grid](
                grad_spike_seq.contiguous(),
                py_dict['grad_v_init'],
                grad_v_seq,
                py_dict['grad_x_seq'],
                py_dict['h_seq'],
                py_dict['v_th'],
                py_dict['v_reset'],
                py_dict['detach_reset'],
                py_dict['numel'],
                py_dict['N'],
                py_dict['T'],
                surrogate_function=map[surrogate_function.__class__.__name__],
                alpha=alpha,
                beta=beta,
                leak=leak,
                k=k,
                w=w,
                c=c,
            )

        return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None, None, None, None

# -----------------------
# IFNode_multi_step_kernels
# -----------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048},     num_warps=16, num_stages=8, ),
        triton.Config({'BLOCK_SIZE': 1024},     num_warps=8,  num_stages=4, ),
        triton.Config({'BLOCK_SIZE': 512},      num_warps=4,  num_stages=2, ),
        triton.Config({'BLOCK_SIZE': 256},      num_warps=4,  num_stages=2, ),
    ],
    key=['N'],
)
@triton.jit
def IFNode_multi_step_forward_kernel(
        x_seq_ptr , # [T, N]
        v_v_seq_ptr, # [T + 1, N]
        h_seq_ptr, # [T, N]
        spike_seq_ptr, # [T, N]
        v_th, 
        v_reset, 
        numel,
        N: tl.constexpr,
        T: tl.constexpr,
        BLOCK_SIZE:tl.constexpr,
    ):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    for t in tl.static_range(0, T):
        offset = t * N + offset_n
        mask = offset < numel
        x_seq_t = tl.load(x_seq_ptr + offset, mask=mask)
        v_v_seq_t = tl.load(v_v_seq_ptr + offset, mask=mask)
        
        h_seq_t = x_seq_t + v_v_seq_t
        tl.store(h_seq_ptr + offset, h_seq_t)
        spike_seq_t = h_seq_t >= v_th
        tl.store(spike_seq_ptr + offset, spike_seq_t)

        if v_reset is not None: # hard reset
            v_v_seq_t_plus_dt = h_seq_t * (1. - spike_seq_t) + v_reset * spike_seq_t
        else: 
            v_v_seq_t_plus_dt = h_seq_t - spike_seq_t * v_th
        
        tl.store(v_v_seq_ptr + offset + N, v_v_seq_t_plus_dt)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048},     num_warps=16,  num_stages=4, ),
        triton.Config({'BLOCK_SIZE': 1024},     num_warps=8,  num_stages=4, ),
        triton.Config({'BLOCK_SIZE': 512},      num_warps=4,  num_stages=2, ),
        triton.Config({'BLOCK_SIZE': 256},      num_warps=4,  num_stages=2, ),
    ],  
    key=['N'],

)    
@triton.jit
def IFNode_multi_step_backward_kernel(
    grad_spike_seq_ptr, # [T, N]
    grad_v_init_ptr, # [N] ,output
    grad_v_seq_ptr, # [T, N]
    grad_x_seq_ptr, # [T, N] ,output
    h_seq_ptr, # [T, N]
    v_th,
    v_reset,
    detach_reset,
    numel,
    N: tl.constexpr, # num of col in grad_spike_seq
    T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # *args,
    surrogate_function: tl.constexpr,
    alpha,
    beta,
    leak,
    k,
    w,
    c,
):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    grad_h = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for t in tl.static_range(T - 1, -1, -1):
        offset = t * N + offset_n
        mask = offset < numel
        h_seq = tl.load(h_seq_ptr + offset, mask=mask)
        grad_v_seq = tl.load(grad_v_seq_ptr + offset, mask=mask)
        grad_spike_seq = tl.load(grad_spike_seq_ptr + offset, mask=mask)

        over_th = h_seq - v_th
        spike_seq = (over_th >= 0).to(h_seq.dtype)

         # ATan
        if surrogate_function == 1: 
            grad_s_to_h = atan_grad_s_to_h(over_th, alpha, h_seq.dtype)
        # FakeNumericalGradient
        elif surrogate_function == 2:
            grad_s_to_h = fakeNumericalGradient_grad_s_to_h(over_th, alpha)
        # LeakyKReLU
        elif surrogate_function == 3:
            grad_s_to_h = leakyKReLU_grad_s_to_h(spike_seq, leak, k)
        # PiecewiseLeakyReLU
        elif surrogate_function == 4: 
            grad_s_to_h = piecewiseLeakyReLU_grad_s_to_h(over_th, spike_seq, w, c)
        # QPseudoSpike
        elif surrogate_function == 5: 
            grad_s_to_h = qPseudoSpike_grad_s_to_h(over_th, spike_seq, alpha)
        # Sigmoid
        elif surrogate_function == 6: 
            grad_s_to_h = sigmoid_grad_s_to_h(over_th, alpha)
        # S2NN
        elif surrogate_function == 7: 
            grad_s_to_h = s2NN_grad_s_to_h(over_th, spike_seq, alpha, beta)

        if detach_reset:
            if v_reset is not None: # hard reset
                grad_v_to_h = (1. - spike_seq).to(spike_seq.dtype)
            else:
                grad_v_to_h = tl.full(spike_seq.shape, 1, dtype=spike_seq.dtype)
        else:
            if v_reset is not  None:# hard reset
                grad_v_to_h = (1. - spike_seq + (v_reset - h_seq) * grad_s_to_h).to(spike_seq.dtype)
            else:
                grad_v_to_h = (1. - v_th * grad_s_to_h).to(spike_seq.dtype)

        grad_h = grad_spike_seq * grad_s_to_h + (grad_v_seq + grad_h) * grad_v_to_h
        grad_x_seq = grad_h
        tl.store(grad_x_seq_ptr + offset, grad_x_seq)
    
    grad_v_init = grad_h
    tl.store(grad_v_init_ptr + offset_n, grad_v_init)

# -----------------------
# IFNode_single_step_kernels
# -----------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048},     num_warps=16,  num_stages=4, ),
        triton.Config({'BLOCK_SIZE': 1024},     num_warps=8,  num_stages=4, ),
        triton.Config({'BLOCK_SIZE': 512},      num_warps=4,  num_stages=2, ),
        triton.Config({'BLOCK_SIZE': 256},      num_warps=4,  num_stages=2, ),
    ],  
    key=['N'],
)
@triton.jit
def IFNode_single_step_forward_kernel(
        x_ptr, # [N] Input
        v_ptr, # [N] Input
        h_ptr, # [N] Output
        spike_ptr, # [N] Output
        v_next_ptr, # [N] Output
        v_th,
        v_reset,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offset < N
    x = tl.load(x_ptr + offset, mask=mask)
    v = tl.load(v_ptr + offset, mask=mask)
    
    h = x + v
    tl.store(h_ptr + offset, h, mask=mask)
    spike = (h >= v_th)
    tl.store(spike_ptr + offset, spike, mask=mask)

    if v_reset is not None: # hard reset
        v_plus_dt = h * (1. - spike) + v_reset * spike
    else: # soft reset
        v_plus_dt = h - spike * v_th

    tl.store(v_next_ptr + offset, v_plus_dt, mask=mask)
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048},     num_warps=16,  num_stages=8, ),
        triton.Config({'BLOCK_SIZE': 1024},     num_warps=8,  num_stages=4, ),
        triton.Config({'BLOCK_SIZE': 512},      num_warps=4,  num_stages=2, ),
        triton.Config({'BLOCK_SIZE': 256},      num_warps=4,  num_stages=2, ),
    ],
    key=['N'],
)
@triton.jit
def IFNode_single_step_backward_kernel(
    grad_spike_ptr, # [N] Input
    grad_v_ptr, # [N] Input
    h_ptr, # [N] Input
    grad_x_ptr, # [N] output
    grad_v_next_ptr, # [N] output
    v_th,
    v_reset,
    detach_reset,
    N: tl.constexpr,  
    BLOCK_SIZE: tl.constexpr,
    surrogate_function: tl.constexpr,
    # *args,
    alpha,
    beta,
    leak,
    k,
    w,
    c,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offset < N
    h = tl.load(h_ptr + offset, mask=mask)
    grad_v = tl.load(grad_v_ptr + offset, mask=mask)
    grad_spike = tl.load(grad_spike_ptr + offset, mask=mask)
    
    over_th = h - v_th
    spike = (over_th >= 0).to(grad_spike.dtype)
    # ATan
    if surrogate_function == 1: 
        grad_s_to_h = atan_grad_s_to_h(over_th, alpha, h.dtype)
    # FakeNumericalGradient
    elif surrogate_function == 2:
        grad_s_to_h = fakeNumericalGradient_grad_s_to_h(over_th, alpha)
     # LeakyKReLU
    elif surrogate_function == 3:
        grad_s_to_h = leakyKReLU_grad_s_to_h(spike, leak, k)
    # PiecewiseLeakyReLU
    elif surrogate_function == 4: 
        grad_s_to_h = piecewiseLeakyReLU_grad_s_to_h(over_th, spike, w, c)
    # QPseudoSpike
    elif surrogate_function == 5: 
        grad_s_to_h = qPseudoSpike_grad_s_to_h(over_th, spike, alpha)
    # Sigmoid
    elif surrogate_function == 6: 
        grad_s_to_h = sigmoid_grad_s_to_h(over_th, alpha)
    # S2NN
    elif surrogate_function == 7: 
        grad_s_to_h = s2NN_grad_s_to_h(over_th, spike, alpha, beta)

    if detach_reset:
        if v_reset is not None:
            grad_v_to_h = (1. - spike).to(spike.dtype)
        else:
            grad_v_to_h = tl.full(spike.shape, 1, dtype=spike.dtype)
    else:
        if v_reset is not  None:
            grad_v_to_h = (1. - spike + (v_reset - tl.load(h_ptr + offset)) * grad_s_to_h).to(spike.dtype)
        else:
            grad_v_to_h = (1. - v_th * grad_s_to_h).to(spike.dtype)

    grad_h = grad_spike * grad_s_to_h + grad_v  * grad_v_to_h
    grad_x = grad_h

    tl.store(grad_x_ptr + offset, grad_x)
    tl.store(grad_v_next_ptr + offset, grad_h)   