import torch
import torch.nn as nn
from typing import Callable, Union
import spikingjelly.activation_based.surrogate as surrogate
# from . import neuron_backend
import neuron_backend

class IFNode(nn.Module):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode = 'm',
                 backend = 'torch', store_v_seq: bool = False):
        super().__init__()
        self.v_threshold = v_threshold
        
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

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def neuronal_fire(self):
        spike = self.surrogate_function(self.v - self.v_threshold)
        return spike
    
    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)
    

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
        v = v + x
        spike = (v >= v_threshold).to(x) 
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                               v_reset: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                          v_reset: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq


    def torch_single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
    
    def torch_multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)   

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        if self.training:
            if self.backend == 'torch':
                return self.torch_multi_step_forward(x_seq)
            elif self.backend == 'triton':
                forward_kernel, backward_kernel = neuron_backend.IFNode_multi_step_forward_kernel, neuron_backend.IFNode_multi_step_backward_kernel

                self.v_float_to_tensor(x_seq[0])

                spike_seq, v_seq = neuron_backend.IFNodeMultiStepATGF.apply(
                                                    x_seq.flatten(1), 
                                                    self.v.flatten(0),
                                                    self.v_threshold, self.v_reset,
                                                    self.detach_reset,
                                                    self.surrogate_function,
                                                    forward_kernel,
                                                    backward_kernel,
                                                    )
                
                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)

                if self.store_v_seq:
                    self.v_seq = v_seq

                self.v = v_seq[-1].clone()
                return spike_seq
            else:
                raise ValueError(self.backend) 
        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq,
                                                                                                           self.v,
                                                                                                           self.v_threshold)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset(x_seq, self.v, self.v_threshold)
            else:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq,
                                                                                                           self.v,
                                                                                                           self.v_threshold,
                                                                                                           self.v_reset)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset(x_seq, self.v, self.v_threshold,
                                                                                    self.v_reset)
            return spike_seq       
   
    def single_step_forward(self, x: torch.Tensor,*args, **kwargs):
        if self.training:
            if self.backend == 'torch':
                return self.torch_single_step_forward(x)
            elif self.backend == 'triton':
                forward_kernel, backward_kernel =  neuron_backend.IFNode_single_step_forward_kernel, neuron_backend.IFNode_single_step_backward_kernel

                self.v_float_to_tensor(x)

                spike, v = neuron_backend.IFNodeSingleStepATGF.apply(
                                                x.flatten(0),
                                                self.v.flatten(0),
                                                self.v_th, self.v_reset,
                                                self.detach_reset,
                                                self.surrogate_function,
                                                forward_kernel,
                                                backward_kernel,
                                                )
                spike = spike.reshape(x.shape)
                v = v.reshape(x.shape)
                
                self.v = v
                return spike
            else:
                raise ValueError(self.backend)
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                spike, self.v = self.jit_eval_single_step_forward_soft_reset(x, self.v, self.v_threshold)
            else:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset(x, self.v, self.v_threshold, self.v_reset)
            return spike
    
    def forward(self, *args, **kwargs):
        if self.step_mode == 's':
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)       


class LinearNode(nn.Module):
    def __init__(self,a: float, b:float, learnable: bool = False, v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode = 'm',
                 backend='torch', store_v_seq: bool = False):
        super().__init__()

        self.v_threshold = v_threshold
        if v_reset is None:
            self.v = 0.
        else:
            self.v = v_reset

        self.v_reset = v_reset
        self.detach_reset = detach_reset 
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq

        if learnable:
            self.a = nn.Parameter(torch.as_tensor(a))
            self.b = nn.Parameter(torch.as_tensor(b))
        else:
            self.a = a
            self.b = b
    
    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init) 
    
    def reset(self):
        if self.v_reset is None:
            self.v = 0.
        else:
            self.v = self.v_reset

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.a * self.v + self.b * x

    def neruonal_fire(self):
        spike = self.surrogate_function(self.v - self.v_threshold)
        return spike
    
    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset) 

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float,
                                                a: float, b: float):
        v = a * v + b * x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                a: float, b: float):
        v = a * v + b * x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                               v_reset: float, a: float, b: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = a * v + b * x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                  v_reset: float, a: float, b: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = a * v + b * x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                               a: float, b: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = a * v + b * x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v
    
    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                        a: float, b: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = a * v + b * x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    def torch_single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neruonal_fire()
        self.neuronal_reset(spike)
        return spike
    
    def torch_multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def single_step_forward(self, x: torch.Tensor,*args, **kwargs):
        if self.training:
            if self.backend == 'torch':
                return self.torch_single_step_forward(x)
            elif self.backend == 'triton':
                pass
            else:
                raise ValueError(self.backend)
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                spike, self.v = self.jit_eval_single_step_forward_soft_reset(x, self.v, self.v_threshold)
            else:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset(x, self.v, self.v_threshold, self.v_reset)
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        if self.training:
            if self.backend == 'torch':
                return self.torch_multi_step_forward(x_seq)
            elif self.backend == 'triton':
                forward_kernel, backward_kernel = neuron_backend.LinearNode_multi_step_forward_kernel, neuron_backend.LinearNode_multi_step_backward_kernel

                self.v_float_to_tensor(x_seq[0])
            else:
                raise ValueError(self.backend)
        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq,self.v,self.v_threshold,
                                                                                                           self.a,self.b)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset(x_seq, self.v, self.v_threshold,
                                                                                    self.a, self.b)
            else:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq,self.v,self.v_threshold,self.v_reset,
                                                                                                           self.a,self.b)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset(x_seq, self.v, self.v_threshold,
                                                                                    self.v_reset, self.a, self.b)
            return spike_seq

    def forward(self, *args, **kwargs):
        if self.step_mode == 's':
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)       


@torch.no_grad()
def max_error(x: torch.Tensor, y: torch.Tensor):
    return (x - y).abs().max()

from spikingjelly.activation_based import neuron,cuda_utils, functional

def forward_backward(net: torch.nn.Module, x_seq: torch.Tensor):
    y_seq = net(x_seq)
    y_seq.sum().backward()
    x_seq.grad.zero_()
    functional.reset_net(net)

if __name__ == '__main__':
    T = 8
    N = 64
    C = 32 * 32 * 32
    device = 'cuda:2'
    # for i in range(10):
    #     for surrogate_function in surrogate._has_cuda_:
    #         print(f'surrogate_function = {surrogate_function}')
    #         net_triton = IFNode(backend='triton', step_mode='m', surrogate_function=surrogate_function(), detach_reset=True)
    #         # net_cupy = neuron.IFNode(backend='torch', step_mode='m', surrogate_function=surrogate_function(), detach_reset=True)

    #         for dtype in [torch.float32]:
    #             # torch.manual_seed(0)
    #             x_seq = torch.rand([T, N, C], device=device, requires_grad=True, dtype=dtype)

    #             y_triton = net_triton(x_seq)
    #             y_triton.sum().backward()
    #             x_grad_triton = x_seq.grad.clone()
    #             x_seq.grad.zero_()
    #             functional.reset_net(net_triton)

    #             print(f'dtype = {dtype}, max error of x_seq.grad = {max_error(x_grad_triton, x_grad_triton)}')
    with torch.cuda.device(device):
        x_seq = torch.rand([T, N, C], device=device, requires_grad=True, dtype=torch.float32)
        net_IFNode = IFNode(backend='torch',surrogate_function=surrogate.Sigmoid(), step_mode='m', detach_reset=True)
        net_LinearNode = LinearNode(a=1.0, b=1.0, learnable=True,  backend='torch', surrogate_function=surrogate.Sigmoid(), step_mode='m', detach_reset=True)

        y_IFNode = net_IFNode(x_seq)
        y_IFNode.sum().backward()
        x_grad_IFNode = x_seq.grad.clone()
        x_seq.grad.zero_()

        y_LinearNode = net_LinearNode(x_seq)
        y_LinearNode.sum().backward()
        x_grad_LinearNode = x_seq.grad.clone()
        x_seq.grad.zero_()
        
        print(f'max error of y_IFNode = {max_error(y_IFNode, y_LinearNode)}')
        print(f'max error of x_seq.grad = {max_error(x_grad_IFNode, x_grad_LinearNode)}')

    # print(f'dtype = {dtype}, max_error of y_seq =  {max_error(y_triton, y_cupy)}')
    # print(f'dtype = {dtype}, max error of x_seq.grad = {max_error(x_grad_triton, x_grad_cupy)}')

# if __name__ == '__main__':
#     N = 64
#     device = 'cuda:0'

#     repeats = 32

#     for surrogate_function in surrogate._has_cuda_:
#         print(f'surrogate_function = {surrogate_function}')
#         net_triton = IFNode(backend='triton',surrogate_function=surrogate_function(), detach_reset=True)
#         net_cupy = neuron.IFNode(backend='cupy', surrogate_function=surrogate_function(), step_mode='m', detach_reset=True)

#         for dtype in [torch.half, torch.float]:
#             for C in [32 * 32, 32 * 32  * 32, 32 * 32 * 32 * 4, 32 * 32 * 32 * 8]:
#                 print('N * C = ', N * C)
#                 for T in [2, 4, 8, 16, 32]:
#                     x_seq = torch.rand([T, N, C], device=device, requires_grad=True, dtype=torch.float16)

#                     t_cupy = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_cupy, x_seq)
#                     t_triton = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_triton, x_seq)

#                     print(f'dtype={dtype}, T={T},'.ljust(30), f't_cupy / t_triton = {round(t_cupy / t_triton, 2)}')