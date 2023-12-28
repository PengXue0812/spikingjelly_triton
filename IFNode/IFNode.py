import torch
import torch.nn as nn
from typing import Callable
import copy
import triton
import triton.language as tl
import spikingjelly.activation_based.surrogate as surrogate


def if_requries_grad(items):
    for item in items:
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                return True
    return False
                          
def new_tensors(news: tuple, py_dict: dict, ref: str='x_seq'):
    ref = py_dict[ref]
    zero_shape = list(ref.shape)
    zero_shape[0] *= news.__len__()
    for i, item in enumerate(torch.split(torch.zeros(zero_shape, device=ref.device, dtype=ref.dtype), ref.shape[0])):
        py_dict[news[i]] = item

def preforward(py_dict):
    requires_grad = if_requries_grad(py_dict.values())
    
    new_tensors(('h_seq', 'spike_seq', 'v_seq'), py_dict)
    py_dict['v_v_seq'] = torch.cat((py_dict.pop('v_init').unsqueeze(0), py_dict.pop('v_seq')))
    numel = py_dict['x_seq'].numel()
    N = py_dict['x_seq'].shape[1]

    threads = 4096  
    BLOCK_SIZE = 4096 
    blocks = (N + threads - 1) // threads // BLOCK_SIZE
    
    grid = (blocks,threads,)
    py_dict['numel'] = numel
    py_dict['N'] = N
    py_dict['T'] = py_dict['x_seq'].shape[0]
    py_dict['BLOCK_SIZE'] = BLOCK_SIZE

    return requires_grad, grid, py_dict

def prebackward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
    backward_kernel = ctx.backward_kernel
    grid = ctx.grid

    numel = ctx.numel
    h_seq = ctx.h_seq
    N = ctx.N
    T = ctx.T
    BLOCK_SIZE = ctx.BLOCK_SIZE
    v_th = ctx.v_th
    v_reset = ctx.v_reset

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
        'numel': numel,
        'N': N,
        'T': T,
        'BLOCK_SIZE': BLOCK_SIZE
    }
    return backward_kernel, grid, py_dict

def ctx_save(ctx, requires_grad: bool, *args, **kwargs):
    if requires_grad:
        ctx.save_for_backward(*args)
        for key, value in kwargs.items():
            ctx.__setattr__(key, value)

class IFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None,
                forward_kernel, backward_kernel):
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset
        }

        requires_grad, grid, py_dict = preforward(py_dict)

        if py_dict['v_reset'] is None: # soft-reset
            py_dict.pop('v_reset')

        forward_kernel[grid](
            x_seq,
            py_dict['v_v_seq'],
            py_dict['h_seq'],
            py_dict['spike_seq'],
            py_dict['v_th'],
            py_dict['v_reset'],
            py_dict['numel'],
            py_dict['N'],
            py_dict['T'],
            BLOCK_SIZE=py_dict['BLOCK_SIZE']
         )
        
        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        ctx_save(ctx, requires_grad, h_seq=py_dict['h_seq'], grid=grid, 
                            BLOCK_SIZE=py_dict['BLOCK_SIZE'], numel=py_dict['numel'],
                            N=py_dict['N'], T=py_dict['T'],
                            v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                            backward_kernel=backward_kernel)
        return py_dict['spike_seq'], py_dict['v_v_seq'][1:,]
    
    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
    
        backward_kernel, grid, py_dict = prebackward(ctx, grad_spike_seq, grad_v_seq)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')
        
        backward_kernel[grid](
            grad_spike_seq.contiguous(),
            py_dict['grad_v_init'],
            grad_v_seq,
            py_dict['grad_x_seq'],
            py_dict['h_seq'],
            py_dict['v_th'],
            py_dict['v_reset'],
            py_dict['numel'],
            py_dict['N'],
            py_dict['T'],
            BLOCK_SIZE=py_dict['BLOCK_SIZE']
        )

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None
        
        return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None

@triton.jit
def triton_multi_step_forward_hard_reset(
        x_seq_ptr , # [T, N]
        v_v_seq_ptr, # [T + 1, N]
        h_seq_ptr, # [T, N]
        spike_seq_ptr, # [T, N]
        v_th, 
        v_reset, 
        numel,
        N: tl.constexpr,
        T:tl.constexpr,
        BLOCK_SIZE:tl.constexpr,
    ):
    block = tl.program_id(axis=0)
    thread = tl.program_id(axis=1)
    pid = block * tl.num_programs(1) + thread
    
    offset_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # for block_idx in range(0, BLOCK_SIZE):
    for t in tl.static_range(0, T):
        offset = t * N + offset_n

        mask = offset < numel
        x_seq_t = tl.load(x_seq_ptr + offset, mask=mask)
        v_v_seq_t = tl.load(v_v_seq_ptr + offset, mask=mask)

        h_seq_t = x_seq_t + v_v_seq_t
        spike_seq_t = h_seq_t >= v_th

        v_v_seq_t_plus_dt = h_seq_t * (1. - spike_seq_t) + v_reset * spike_seq_t
        tl.store(h_seq_ptr + offset, h_seq_t, mask=mask)
        tl.store(spike_seq_ptr + offset, spike_seq_t, mask=mask)
        mask2 = offset < numel + N
        tl.store(v_v_seq_ptr + offset + N, v_v_seq_t_plus_dt, mask=mask2)
        
@triton.jit
def triton_multi_step_backward_hard_reset(
    grad_spike_seq_ptr, # [T, N]
    grad_v_init_ptr, # [N] ,output
    grad_v_seq_ptr, # [T, N]
    grad_x_seq_ptr, # [T, N] ,output
    h_seq_ptr, # [T, N]
    v_th,
    v_reset,
    numel,
    N: tl.constexpr, # num of col in grad_spike_seq
    T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(axis=0)
    thread = tl.program_id(axis=1)
    pid = block * tl.num_programs(1) + thread

    offset_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    grad_h = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for t in tl.static_range(T - 1, -1, -1):
        offset = t * N + offset_n
        mask = offset < numel
        h_seq = tl.load(h_seq_ptr + offset, mask=mask)
        grad_v_seq = tl.load(grad_v_seq_ptr + offset, mask=mask)
        grad_spike_seq = tl.load(grad_spike_seq_ptr + offset, mask=mask)

        over_th = h_seq - v_th
        spike_seq = (over_th >= 0)
        sigmoid_backward__sigmoid_ax = 1. / (1. + tl.math.exp(-4. * over_th))
        grad_s_to_h = (1. - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * 4.
        grad_v_to_h = 1. - spike_seq
        temp_var = v_reset - h_seq
        temp_var = temp_var * grad_s_to_h
        grad_v_to_h = temp_var + grad_v_to_h
        grad_h_next_to_v = 1.
        grad_h = grad_h * grad_h_next_to_v
        grad_h = grad_v_seq + grad_h
        grad_h = grad_h * grad_v_to_h
        temp_var = grad_spike_seq * grad_s_to_h
        grad_h = grad_h + temp_var

        grad_h_to_x = 1.
        grad_x_seq = grad_h * grad_h_to_x
        tl.store(grad_x_seq_ptr + offset, grad_x_seq, mask=mask)
    
    grad_h_next_to_v = 1.
    grad_v_init = grad_h * grad_h_next_to_v
    mask = offset_n < numel
    tl.store(grad_v_init_ptr + offset_n, grad_v_init, mask=mask)

    
class IFNode(nn.Module):
    def __init__(self, v_th: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode = 'm',
                 backend = 'triton', store_v_seq: bool = False):
        super().__init__()
        self.v_th = v_th
        self._memories = {}
        self._memories_rv = {}

        if v_reset is None:
            self.v = 0.

        else:
            self.v = v_reset

        self.v_reset = v_reset
        self.detach_reset = detach_reset # 是否将reset过程的计算图分离
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init) 
    
    def reset(self):
        if self.v_reset is None:
            self.v = 0.
        else:
            self.v = self.v_reset

    def forward(self, x_seq: torch.Tensor):
        self.v_float_to_tensor(x_seq[0])

        spike_seq, v_seq = IFNodeATGF.apply(x_seq.flatten(1), 
                                            self.v.flatten(0),
                                            self.v_th, self.v_reset,
                                            triton_multi_step_forward_hard_reset,
                                            triton_multi_step_backward_hard_reset)
        
        spike_seq = spike_seq.reshape(x_seq.shape)
        v_seq = v_seq.reshape(x_seq.shape)

        if self.store_v_seq:
            self.v_seq = v_seq

        self.v = v_seq[-1].clone()
        return spike_seq
      
@torch.no_grad()
def max_error(x: torch.Tensor, y: torch.Tensor):
    return (x - y).abs().max()


from spikingjelly.activation_based import neuron, cuda_utils, functional

def forward_backward(net: torch.nn.Module, x_seq: torch.Tensor):
    y_seq = net(x_seq)
    y_seq.sum().backward()
    x_seq.grad.zero_()
    functional.reset_net(net)

if __name__ == '__main__':
    N = 64
    C = 32 * 32 * 32 
    device = 'cuda:0'

    repeats = 16

    net_triton = IFNode()
    net_torch = neuron.IFNode(backend='torch', step_mode='m')
    net_cupy = neuron.IFNode(backend='cupy', step_mode='m')

    for T in [2, 4, 8, 16, 32, 64]:
        x_seq = torch.rand([T, N, C], device=device, requires_grad=True, dtype=torch.float32)

        t_triton = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_triton, x_seq)
        t_torch = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_torch, x_seq)
        t_cupy = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_cupy, x_seq)

        # print(f'T={T},'.ljust(30), f't_torch / t_triton = {round(t_torch / t_triton, 2)}')
        print(f'T={T},'.ljust(30), f't_cupy / t_triton = {round(t_cupy / t_triton, 2)}')

    # T = 8
    # N = 64
    # C = 32 * 32 * 32
    # device = 'cuda:0'
    # for T in [2, 4, 8, 16]:
    #     x_seq = torch.rand([T, N, C], device=device, requires_grad=True)

    #     net_triton = IFNode()
    #     y_triton = net_triton(x_seq)
    #     y_triton.sum().backward()
    #     x_grad_triton = x_seq.grad.clone()
    #     x_seq.grad.zero_()

    #     net_torch = neuron.IFNode(backend='torch', step_mode='m')
    #     y_torch = net_torch(x_seq)
    #     y_torch.sum().backward()
    #     x_grad_torch = x_seq.grad.clone()

    #     print('max error of y_seq', max_error(y_triton, y_torch))
    #     print('max error of x_seq.grad', max_error(x_grad_triton, x_grad_torch))