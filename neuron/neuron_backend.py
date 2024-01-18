import torch
import triton
import triton.language as tl

map = {'ATan':1, 'FakeNumericalGradient':2, 'LeakyKReLU':3, 'PiecewiseLeakyReLU':4, 'QPseudoSpike':5, 'Sigmoid':6, 'S2NN':7}

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

def new_tensors(news: tuple, py_dict: dict, ref: str):
    # ref: x if single step, x_seq if multi step
    ref = py_dict[ref]
    zero_shape = list(ref.shape)
    zero_shape[0] *= news.__len__()
    for i, item in enumerate(torch.split(torch.zeros(zero_shape, device=ref.device, dtype=ref.dtype), ref.shape[0])):
        py_dict[news[i]] = item

def pre_single_step_forward(py_dict):
    requires_grad = if_requries_grad(py_dict.values())

    new_tensors(('h_ptr', 'spike_ptr', 'v_next_ptr'), py_dict, ref='x_ptr')
    N = py_dict['x_ptr'].numel()
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    py_dict['N'] = N

    return requires_grad, grid, py_dict

def pre_single_step_backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
    backward = ctx.backward
    grid = ctx.grid

    h = ctx.saved_tensors[0]
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
        'grad_spike_ptr': grad_spike.contiguous(),
        'grad_v_ptr': grad_v,
        'grad_x_ptr': grad_x,
        'grad_v_next_ptr': grad_v_next,
        'h_ptr': h,
        'v_th': v_th,
        'v_reset': v_reset,
        'detach_reset': detach_reset,
        'N': N,
        'surrogate_function': map[surrogate_function.__class__.__name__],
    }

    attributes = ['alpha', 'beta', 'leak', 'k', 'w', 'c']
    for attribute in attributes:
        py_dict[attribute] = getattr(surrogate_function, attribute) if hasattr(surrogate_function, attribute) else None

    return backward, grid, py_dict

def pre_multi_step_forward(py_dict):
    requires_grad = if_requries_grad(py_dict.values())
    
    new_tensors(('h_seq_ptr', 'spike_seq_ptr', 'v_seq_ptr'), py_dict, ref='x_seq_ptr')
    py_dict['v_v_seq_ptr'] = torch.cat((py_dict.pop('v_init_ptr').unsqueeze(0), py_dict.pop('v_seq_ptr')))
    numel = py_dict['x_seq_ptr'].numel()
    N = py_dict['x_seq_ptr'].shape[1]

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )

    py_dict['numel'] = numel
    py_dict['N'] = N
    py_dict['T'] = py_dict['x_seq_ptr'].shape[0]

    return requires_grad, grid, py_dict

def pre_multi_step_backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
    backward = ctx.backward
    grid = ctx.grid

    h_seq = ctx.saved_tensors[0]
    numel = ctx.numel
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
        'grad_spike_seq_ptr': grad_spike_seq.contiguous(),
        'grad_v_seq_ptr': grad_v_seq,
        'grad_x_seq_ptr': grad_x_seq,
        'grad_v_init_ptr': grad_v_init,
        'h_seq_ptr': h_seq,
        'v_th': v_th,
        'v_reset': v_reset,
        'detach_reset': detach_reset,
        'numel': numel,
        'N': N,
        'T': T,
        'surrogate_function': map[surrogate_function.__class__.__name__],
    }

    attributes = ['alpha', 'beta', 'leak', 'k', 'w', 'c']
    for attribute in attributes:
        py_dict[attribute] = getattr(surrogate_function, attribute) if hasattr(surrogate_function, attribute) else None

    return backward, grid, py_dict

@triton.jit
def atan_grad_s_to_h(over_th, alpha):
    sg_ATan_M_PI_2__alpha__x = (1.57079632679489661923) * alpha * over_th
    grad_s_to_h = alpha / 2. / (1. + sg_ATan_M_PI_2__alpha__x * sg_ATan_M_PI_2__alpha__x)
    return grad_s_to_h

@triton.jit
def fakeNumericalGradient_grad_s_to_h(over_th, alpha):
    sg_FakeNumericalGradient_sign = (over_th >= 0.) * 2. - 1.
    grad_s_to_h = tl.math.min(sg_FakeNumericalGradient_sign / over_th, alpha)
    return grad_s_to_h

@triton.jit
def leakyKReLU_grad_s_to_h(spike_seq, leak, k):
    sg_LeakyKReLU_mask1 = spike_seq
    grad_s_to_h = leak * (1. - sg_LeakyKReLU_mask1) + k * sg_LeakyKReLU_mask1
    return grad_s_to_h

@triton.jit
def piecewiseLeakyReLU_grad_s_to_h(over_th, w, c):
    sg_PiecewiseLeakyReLU_x_abs = tl.abs(over_th)
    grad_s_to_h = tl.where(sg_PiecewiseLeakyReLU_x_abs > w, c, 1. / w)
    return grad_s_to_h

@triton.jit
def qPseudoSpike_grad_s_to_h(over_th, alpha):
    sg_QPseudoSpike_base = (1. + 2. / (alpha - 1.) * tl.abs(over_th))
    grad_s_to_h = tl.math.pow(sg_QPseudoSpike_base, -alpha)
    return grad_s_to_h

@triton.jit
def sigmoid_grad_s_to_h(over_th, alpha):
    sigmoid_backward__sigmoid_ax = 1. / (1. + tl.exp(-alpha * over_th))
    grad_s_to_h = (1. - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * alpha
    return grad_s_to_h

@triton.jit
def s2NN_grad_s_to_h(over_th, alpha, beta):
    sg_S2NN_sigmoid_ax = 1. / (1. + tl.exp(-alpha * over_th))
    sg_S2NN_mask_l = over_th < 0.
    grad_s_to_h =  (1. - sg_S2NN_sigmoid_ax) * sg_S2NN_sigmoid_ax * alpha * sg_S2NN_mask_l + beta / (over_th + 1.00005) * (1. - sg_S2NN_mask_l) 
    return grad_s_to_h

@triton.jit
def get_grad_s_to_h(surrogate_function, spike_seq, over_th,  alpha, beta, leak, k, w, c):
    if surrogate_function == 1: # ATan
        return atan_grad_s_to_h(over_th, alpha)
    if surrogate_function == 2: # FakeNumericalGradient
        return fakeNumericalGradient_grad_s_to_h(over_th, alpha)
    if surrogate_function == 3: # LeakyKReLU
        return leakyKReLU_grad_s_to_h(spike_seq, leak, k)
    if surrogate_function == 4: # PiecewiseLeakyReLU
        return piecewiseLeakyReLU_grad_s_to_h(over_th, w, c)
    if surrogate_function == 5: # QPseudoSpike
        return qPseudoSpike_grad_s_to_h(over_th, alpha)
    if surrogate_function == 6: # Sigmoid
        return sigmoid_grad_s_to_h(over_th, alpha)
    if surrogate_function == 7: # S2NN
        return s2NN_grad_s_to_h(over_th, alpha, beta)
    
    raise ValueError('surrogate_function is valid')

@triton.jit
def get_grad_v_to_h(spike_seq, h_seq, grad_s_to_h, detach_reset, v_reset, v_th):
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

    return grad_v_to_h

# ++++++++++++++++++++++++++++++++++++++++++++IFNode++++++++++++++++++++++++++++++++++++++++++++
# -----------------------
# IFNodeATGF
# -----------------------
class IFNodeSingleStepATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, v: torch.Tensor, v_th: float, v_reset: float or None, detach_reset: bool,
                surrogate_function,forward, backward):
        py_dict = {
            'x_ptr': x,
            'v_ptr': v,
            'v_th': v_th,
            'v_reset': v_reset,
        }
        requires_grad, grid, py_dict = pre_single_step_forward(py_dict)

        forward[grid](**py_dict)

        ctx_save(ctx, requires_grad, py_dict['h_ptr'], grid=grid,N=py_dict['N'],
                            v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                            backward=backward,detach_reset=detach_reset,
                            surrogate_function=surrogate_function,
                            )
        
        return py_dict['spike_ptr'], py_dict['v_next_ptr']
    
    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
        backward, grid, py_dict = pre_single_step_backward(ctx, grad_spike, grad_v_next)

        backward[grid](**py_dict)
        
        return py_dict['grad_x_ptr'], py_dict['grad_v_next_ptr'], None, None, None, None, None, None, None

class IFNodeMultiStepATGF(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None, detach_reset: bool,
                surrogate_function, forward, backward):
        py_dict = {
            'x_seq_ptr': x_seq,
            'v_init_ptr': v_init,
            'v_th': v_th,
            'v_reset': v_reset,
        }
        requires_grad, grid, py_dict = pre_multi_step_forward(py_dict)

        forward[grid](**py_dict)

        ctx_save(ctx, requires_grad, py_dict['h_seq_ptr'], grid=grid, 
                            numel=py_dict['numel'], detach_reset=detach_reset,
                            N=py_dict['N'], T=py_dict['T'],
                            v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                            backward=backward,
                            surrogate_function=surrogate_function)
        return py_dict['spike_seq_ptr'], py_dict['v_v_seq_ptr'][1:,]
            
    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward, grid, py_dict = pre_multi_step_backward(ctx, grad_spike_seq, grad_v_seq)

        backward[grid](**py_dict)

        return py_dict['grad_x_seq_ptr'], py_dict['grad_v_init_ptr'], None, None, None, None, None, None

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
    surrogate_function: tl.constexpr,
    # *args,
    alpha, beta, leak, k, w, c,
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
        grad_s_to_h = get_grad_s_to_h(surrogate_function, spike_seq, over_th, alpha, beta, leak, k, w, c)
        grad_v_to_h = get_grad_v_to_h(spike_seq, h_seq, grad_s_to_h, detach_reset, v_reset, v_th)

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
    alpha, beta, leak, k, w, c,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offset < N
    h = tl.load(h_ptr + offset, mask=mask)
    grad_v = tl.load(grad_v_ptr + offset, mask=mask)
    grad_spike = tl.load(grad_spike_ptr + offset, mask=mask)
    
    over_th = h - v_th
    spike = (over_th >= 0).to(grad_spike.dtype)
    grad_s_to_h = get_grad_s_to_h(surrogate_function, spike, over_th, alpha, beta, leak, k, w, c)
    grad_v_to_h = get_grad_v_to_h(spike, h, grad_s_to_h, detach_reset, v_reset, v_th)

    grad_h = grad_spike * grad_s_to_h + grad_v  * grad_v_to_h
    grad_x = grad_h

    tl.store(grad_x_ptr + offset, grad_h)
    tl.store(grad_v_next_ptr + offset, grad_x)   

# ++++++++++++++++++++++++++++++++++++++++++++LinearNode++++++++++++++++++++++++++++++++++++++++++++
# -----------------------
# LinearNodeATGF
# -----------------------
class LinearNodeSingleStepATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, learnable: bool, v_th: float, v_reset: float or None, detach_reset: bool,
                surrogate_function,forward, backward):
        py_dict = {
            'x_ptr': x,
            'v_ptr': v,
            'v_th': v_th,
            'v_reset': v_reset,
            'a_ptr': a,
            'b_ptr': b,
        }
        requires_grad, grid, py_dict = pre_single_step_forward(py_dict)

        forward[grid](**py_dict)

        ctx_save(ctx, requires_grad, py_dict['h_ptr'], py_dict['a_ptr'], py_dict['b_ptr'], py_dict['v_ptr'], py_dict['x_ptr'], 
                            learnable=learnable, grid=grid,  detach_reset=detach_reset, N=py_dict['N'], v_th=py_dict['v_th'],
                            v_reset=py_dict['v_reset'], backward=backward, surrogate_function=surrogate_function,
                            )
        
        return py_dict['spike_ptr'], py_dict['v_next_ptr']
    
    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
        backward, grid, py_dict = pre_single_step_backward(ctx, grad_spike, grad_v_next)

        py_dict['a_ptr'] = ctx.saved_tensors[1]
        py_dict['b_ptr'] = ctx.saved_tensors[2]
        py_dict['v_init_ptr'] = ctx.saved_tensors[3]
        py_dict['x_ptr'] = ctx.saved_tensors[4]
        py_dict['learnable'] = ctx.learnable
        grad_a_ptr = torch.as_tensor(0., device=grad_spike.device, dtype=torch.float32)
        grad_b_ptr = torch.as_tensor(0., device=grad_spike.device, dtype=torch.float32)

        backward[grid](grad_a_ptr, grad_b_ptr, **py_dict)

        return py_dict['grad_x_ptr'], py_dict['grad_v_next_ptr'], grad_a_ptr, grad_b_ptr, None, None, None, None, None, None, None

class LinearNodeMultiStepATGF(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, a: torch.Tensor, b: torch.tensor, learnable, v_th: float, v_reset: float or None, detach_reset: bool,
                surrogate_function, forward, backward):
        py_dict = {
            'x_seq_ptr': x_seq,
            'v_init_ptr': v_init,
            'v_th': v_th,
            'v_reset': v_reset,     
            'a_ptr': a,
            'b_ptr': b,
        }
        requires_grad, grid, py_dict = pre_multi_step_forward(py_dict)

        forward[grid](**py_dict)


        ctx_save(ctx, requires_grad,  py_dict['h_seq_ptr'],  py_dict['a_ptr'], py_dict['b_ptr'], py_dict['v_v_seq_ptr'][0], py_dict['x_seq_ptr'],  
                            learnable=learnable, grid=grid, numel=py_dict['numel'], detach_reset=detach_reset,
                            N=py_dict['N'], T=py_dict['T'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                            backward=backward, surrogate_function=surrogate_function)
        return py_dict['spike_seq_ptr'], py_dict['v_v_seq_ptr'][1:,], 
        
    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward, grid, py_dict = pre_multi_step_backward(ctx, grad_spike_seq, grad_v_seq)

        py_dict['a_ptr'] = ctx.saved_tensors[1]
        py_dict['b_ptr'] = ctx.saved_tensors[2]
        py_dict['v_seq_ptr'] = ctx.saved_tensors[3]
        py_dict['x_seq_ptr'] = ctx.saved_tensors[4]
        py_dict['learnable'] = ctx.learnable
        grad_a_ptr = torch.as_tensor(0., device=grad_spike_seq.device, dtype=torch.float32)
        grad_b_ptr = torch.as_tensor(0., device=grad_spike_seq.device, dtype=torch.float32)

        backward[grid](grad_a_ptr, grad_b_ptr, **py_dict)

        return py_dict['grad_x_seq_ptr'], py_dict['grad_v_init_ptr'], grad_a_ptr, grad_b_ptr, None, None, None, None, None, None, None

# -----------------------
# LinearNode_multi_step_kernels
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
def LinearNode_multi_step_forward_kernel(
        x_seq_ptr , # [T, N]
        v_v_seq_ptr, # [T + 1, N]
        h_seq_ptr, # [T, N]
        spike_seq_ptr, # [T, N]
        a_ptr, # [1] Input
        b_ptr, # [1] Input
        v_th, 
        v_reset, 
        numel,
        N: tl.constexpr,
        T: tl.constexpr,
        BLOCK_SIZE:tl.constexpr,
    ):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)

    for t in tl.static_range(0, T):
        offset = t * N + offset_n
        mask = offset < numel
        x_seq_t = tl.load(x_seq_ptr + offset, mask=mask)
        v_v_seq_t = tl.load(v_v_seq_ptr + offset, mask=mask)
        
        h_seq_t = (a * v_v_seq_t + b * x_seq_t).to(v_v_seq_t.dtype)
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
        triton.Config({'BLOCK_SIZE': 2048},     num_warps=16,  num_stages=8, ),
        triton.Config({'BLOCK_SIZE': 1024},     num_warps=8,  num_stages=4, ),
        triton.Config({'BLOCK_SIZE': 512},      num_warps=4,  num_stages=2, ),
        triton.Config({'BLOCK_SIZE': 256},      num_warps=4,  num_stages=2, ),
    ],
    key=['N'],
    reset_to_zero=['grad_a_ptr', 'grad_b_ptr'],
)
@triton.jit
def LinearNode_multi_step_backward_kernel(
    grad_a_ptr, # [1] Input
    grad_b_ptr, # [1] Input
    grad_spike_seq_ptr, # [T, N]
    grad_v_init_ptr, # [N] ,output
    grad_v_seq_ptr, # [T, N]
    grad_x_seq_ptr, # [T, N] ,output
    h_seq_ptr, # [T, N] 
    a_ptr, # [1] 
    b_ptr, # [1]
    v_seq_ptr, # [T, N]
    x_seq_ptr, # [T, N] 
    learnable,
    v_th,
    v_reset,
    detach_reset,
    numel,
    N: tl.constexpr, # num of col in grad_spike_seq
    T: tl.constexpr,
    surrogate_function: tl.constexpr,
    # *args,
    alpha, beta, leak, k, w, c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    grad_h = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    grad_h_next_to_v = tl.load(a_ptr)
    grad_h_to_x = tl.load(b_ptr)
    grad_a = 0.
    grad_b = 0.

    for t in tl.static_range(T - 1, -1, -1):
        offset = t * N + offset_n
        mask = offset < numel
        h_seq = tl.load(h_seq_ptr + offset, mask=mask)
        grad_v_seq = tl.load(grad_v_seq_ptr + offset, mask=mask)
        grad_spike_seq = tl.load(grad_spike_seq_ptr + offset, mask=mask)

        over_th = h_seq - v_th
        spike_seq = over_th >= 0
        grad_s_to_h = get_grad_s_to_h(surrogate_function, spike_seq, over_th, alpha, beta, leak, k, w, c)
        grad_v_to_h = get_grad_v_to_h(spike_seq, h_seq, grad_s_to_h, detach_reset, v_reset, v_th)

        grad_h = grad_spike_seq * grad_s_to_h + (grad_v_seq + grad_h  * grad_h_next_to_v) * grad_v_to_h
        # h = a * v + b * x
        tl.store(grad_x_seq_ptr + offset, grad_h * grad_h_to_x)    

        if learnable:
            x_seq = tl.load(x_seq_ptr + offset, mask=mask)
            v_seq = tl.load(v_seq_ptr + offset, mask=mask)
            grad_a += tl.sum(grad_h * v_seq)
            grad_b += tl.sum(grad_h * x_seq)

    grad_v_init = grad_h * grad_h_next_to_v
    tl.store(grad_v_init_ptr + offset_n, grad_v_init)

    if learnable:
        tl.atomic_add(grad_a_ptr, grad_a)
        tl.atomic_add(grad_b_ptr, grad_b)

# -----------------------
# LinearNode_single_step_kernels
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
def LinearNode_single_step_forward_kernel(
    x_ptr, # [N] Input
    v_ptr, # [N] Input
    h_ptr, # [N] Output
    spike_ptr, # [N] Output
    v_next_ptr, # [N] Output
    a_ptr, # [1] Input
    b_ptr, # [1] Input
    v_th,
    v_reset,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offset < N

    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    x = tl.load(x_ptr + offset, mask=mask)
    v = tl.load(v_ptr + offset, mask=mask)
    
    # h = a * v + b * x
    h = a * v + b * x
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
    reset_to_zero=['grad_a_ptr', 'grad_b_ptr'],
)
@triton.jit
def LinearNode_single_step_backward_kernel(
    grad_a_ptr, # [1] output
    grad_b_ptr, # [1] output
    grad_spike_ptr, # [N] Input
    grad_v_ptr, # [N] Input
    grad_x_ptr, # [N] output
    grad_v_next_ptr, # [N] output
    h_ptr, # [N] Input
    a_ptr, # [1] Input
    b_ptr, # [1] Input
    v_ptr, # [N] Input
    x_ptr, # [N] Input
    learnable,
    v_th,
    v_reset,
    detach_reset,
    N: tl.constexpr,  
    surrogate_function: tl.constexpr,
    # *args,
    alpha, beta, leak, k, w, c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < tl.num_programs(axis=0):
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        mask = offset < N
        h = tl.load(h_ptr + offset, mask=mask)
        grad_v = tl.load(grad_v_ptr + offset, mask=mask)
        grad_spike = tl.load(grad_spike_ptr + offset, mask=mask)
        
        over_th = h - v_th
        spike = (over_th >= 0).to(grad_spike.dtype)
        grad_s_to_h = get_grad_s_to_h(surrogate_function, spike, over_th, alpha, beta, leak, k, w, c)
        grad_v_to_h = get_grad_v_to_h(spike, h, grad_s_to_h, detach_reset, v_reset, v_th)

        grad_h = grad_spike * grad_s_to_h + grad_v * grad_v_to_h

        # h = a * v + b * x
        grad_h_to_v = tl.load(a_ptr)
        grad_h_to_x = tl.load(b_ptr)
        
        tl.store(grad_x_ptr + offset, grad_h * grad_h_to_x)
        tl.store(grad_v_next_ptr + offset, grad_h * grad_h_to_v)   
        
        if learnable:
            v = tl.load(v_ptr + offset, mask=mask)
            x = tl.load(x_ptr + offset, mask=mask)
            grad_a = tl.sum(grad_h * v)
            grad_b = tl.sum(grad_h * x)

            tl.atomic_add(grad_a_ptr, grad_a)
            tl.atomic_add(grad_b_ptr, grad_b)

