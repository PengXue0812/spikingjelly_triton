from abc import abstractmethod
import torch
import torch.nn as nn
from typing import Callable
import spikingjelly.activation_based.surrogate as surrogate
from spikingjelly.activation_based import base
import math
from . import neuron_backend
import torch.nn.functional as F
import numpy as np
# import neuron_backend

# -----------------------
# charge function
# -----------------------
@torch.jit.script
def if_neuronal_charge(x: torch.Tensor, v: torch.Tensor):
    return v + x

@torch.jit.script
def linear_neuronal_charge(x: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    return a * v + b * x

# -----------------------
# fire function
# -----------------------
def neuronal_fire(surrogate_function, x):
    return surrogate_function(x)

# -----------------------
# reset function
# -----------------------
@torch.jit.script
def hard_reset(h: torch.Tensor, spike: torch.Tensor, v_reset: float):
    return (1. - spike) * h + spike * v_reset

@torch.jit.script
def soft_reset(h: torch.Tensor, spike: torch.Tensor, v_threshold: float):
    return h - spike * v_threshold

@torch.jit.script
def neuronal_reset(h: torch.Tensor, spike: torch.Tensor, detach_reset: bool, v_reset: float, v_threshold: float):
    if detach_reset:
        spike_d = spike.detach()
    else:
        spike_d = spike

    if v_reset is None:
        v = soft_reset(h, spike_d, v_threshold)
    else:
        v = hard_reset(h, spike_d, v_reset)
    
    return v


class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_hidden_states: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        
        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)
            
        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        
        self.step_mode = step_mode
        self.backend = backend
        
        self.store_hidden_states = store_hidden_states
    
    @property
    def store_hidden_states(self):
        return self._store_hidden_states

    @store_hidden_states.setter
    def store_hidden_states(self, value: bool):
        self._store_hidden_states = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)
            if not hasattr(self, 'h_seq'):
                self.register_memory('h_seq', None)


    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError
    
    def neuronal_fire(self, h):
        return neuronal_fire(self.surrogate_function, h - self.v_threshold)

    def neuronal_reset(self, h: torch.Tensor, spike: torch.Tensor):
        return neuronal_reset(h, spike, self.detach_reset, self.v_reset, self.v_threshold)
     
    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.h = self.neuronal_charge(x)
        spike = self.neuronal_fire(self.h)
        self.v = self.neuronal_reset(self.h, spike)

        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_hidden_states:
            h_seq = []
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_hidden_states:
                h_seq.append(self.h)
                v_seq.append(self.v)

        if self.store_hidden_states:
            self.h_seq = torch.stack(h_seq)
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)

class IFNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_hidden_states: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_hidden_states)

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', 'triton')
        elif self.step_mode == 'm':
            return ('torch', 'triton')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        return if_neuronal_charge(x, self.v)
    
    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'triton':
            forward_kernel, backward_kernel = neuron_backend.IFNode_multi_step_forward_kernel, neuron_backend.IFNode_multi_step_backward_kernel

            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_backend.IFNodeMultiStepATGF.apply(
                                                x_seq.flatten(1), self.v.flatten(0),
                                                self.training, self.v_threshold,
                                                self.v_reset, self.detach_reset,
                                                self.surrogate_function,
                                                forward_kernel,
                                                backward_kernel,
                                               )
            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_hidden_states:
                self.v_seq = v_seq
            
            self.v = v_seq[-1].clone()
            return spike_seq
          
        
        else:
            raise ValueError(self.backend)
        
    def single_step_forward(self, x: torch.Tensor):
        if self.backend == 'torch':
            return super().single_step_forward(x)
        elif self.backend == 'triton':
            forward_kernel, backward_kernel =  neuron_backend.IFNode_single_step_forward_kernel, neuron_backend.IFNode_single_step_backward_kernel

            self.v_float_to_tensor(x)

            spike, v = neuron_backend.IFNodeSingleStepATGF.apply(
                                            x.flatten(0),self.v.flatten(0),
                                            self.training, self.v_threshold, 
                                            self.v_reset, self.detach_reset,
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

class LinearNode(BaseNode):
    def __init__(self,a: float=1.0, b:float=1.0, learnable: bool = False, v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode = 'm',
                 backend='torch', store_hidden_states: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_hidden_states)
      
        self.learnable = learnable
        self.a = torch.as_tensor(a, dtype=torch.float32)
        self.b = torch.as_tensor(b, dtype=torch.float32)
        if learnable:
            self.a = nn.Parameter(self.a)
            self.b = nn.Parameter(self.b)

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', 'triton')
        elif self.step_mode == 'm':
            return ('torch', 'triton')
        else:
            raise ValueError(self.step_mode)
        
    def extra_repr(self):
        return f'a={self.a}, b={self.b}, learnable={self.learnable}' + super().extra_repr()
    
    def neuronal_charge(self, x: torch.Tensor):
        return linear_neuronal_charge(x, self.v, self.a, self.b)

  
    def single_step_forward(self, x: torch.Tensor):
        if self.backend == 'torch':
            return super().single_step_forward(x)
        elif self.backend == 'triton':
            forward_kernel, backward_kernel = neuron_backend.LinearNode_single_step_forward_kernel, neuron_backend.LinearNode_single_step_backward_kernel

            self.v_float_to_tensor(x)
    
            spike, v = neuron_backend.LinearNodeSingleStepATGF.apply(
                                            x.flatten(0),
                                            self.v.flatten(0),
                                            self.a.to(x.device), self.b.to(x.device),
                                            self.training, self.learnable,
                                            self.v_threshold, self.v_reset,
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

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'triton':
            forward_kernel, backward_kernel = neuron_backend.LinearNode_multi_step_forward_kernel, neuron_backend.LinearNode_multi_step_backward_kernel

            self.v_float_to_tensor(x_seq[0])
            spike_seq, v_seq = neuron_backend.LinearNodeMultiStepATGF.apply(
                                                x_seq.flatten(1),
                                                self.v.flatten(0),
                                                self.a.to(x_seq.device), self.b.to(x_seq.device),
                                                self.training, self.learnable,
                                                self.v_threshold, self.v_reset,
                                                self.detach_reset,
                                                self.surrogate_function,
                                                forward_kernel,
                                                backward_kernel,
                                                )
            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_hidden_states:
                self.v_seq = v_seq
            
            self.v = v_seq[-1].clone()
            return spike_seq
        else:
            raise ValueError(self.backend)

class ParallelNode(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, tau: int, surrogate_function: Callable = surrogate.Sigmoid()):
        super().__init__()
        self.T = T
        self.tau = tau
        self.surrogate_function = surrogate_function
        weight = torch.zeros([T, T])
        bias = torch.zeros([T, 1])

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 
        nn.init.constant_(self.bias, -1) 

    def forward(self, x_seq: torch.Tensor):
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1)) 
        s_sum = torch.zeros_like(h_seq)
        for _ in range(0, self.tau + 1):
            spike_seq = self.surrogate_function(h_seq - s_sum)
            s_sum = s_sum + spike_seq
        return spike_seq.view(x_seq.shape)

        # h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
        # spike_seq_1 = self.surrogate_function(h_seq)
        # spike_seq_2 = self.surrogate_function(h_seq - spike_seq_1)
        # spike_seq = spike_seq_1 + spike_seq_2 - spike_seq_1 * spike_seq_2
        # return spike_seq.view(x_seq.shape)
    
class PSN(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan()):
        super().__init__()
        self.T = T
        self.surrogate_function = surrogate_function
        weight = torch.zeros([T, T])
        bias = torch.zeros([T, 1])

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *].
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq.view(x_seq.shape)

    def extra_repr(self):
        return super().extra_repr() + f'T={self.T}, '

class PSN_OR(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan()):
        super().__init__()
        self.psn0 = PSN(T, surrogate_function)
        self.psn1 = PSN(T, surrogate_function)

    def forward(self, x_seq: torch.Tensor):
        x = self.psn0(x_seq)
        y = self.psn1(x)
        y = x + y - x * y
        return y.view(x_seq.shape)

    def extra_repr(self):
        return super().extra_repr() + f'T={self.T}, '

class DropoutPSN(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan()):
        super().__init__()
        self.T = T
        self.surrogate_function = surrogate_function
        weight = torch.zeros([T, T])
        bias = torch.zeros([T, 1])

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.)


    # @staticmethod
    def drop_time_step_mask(T: int, T_drop: int, device): # 返回一个shape=[T]的mask，其中有T_drop个元素为0，剩余T - T_drop个元素为T / (T - T_drop)
        print("-------------")
        print(T)
        mask = torch.empty([T], device=device)
        nn.init.constant_(mask, T / (T - T_drop))
        mask0 = np.random.choice(T, T_drop, replace=False)
        mask[mask0] = 0.
        return mask

    def forward(self, x_seq: torch.Tensor):
        if self.training:
            mask = DropoutPSN.drop_time_step_mask(self.T, self.T // 4, x_seq.device)
            print(mask)
            weight = self.weight * mask
            h_seq = torch.addmm(self.bias, weight, x_seq.flatten(1))
            spike_seq = self.surrogate_function(h_seq)
            return spike_seq.view(x_seq.shape)
        else:
            h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
            spike_seq = self.surrogate_function(h_seq)
            return spike_seq.view(x_seq.shape)
    

    def extra_repr(self):
        return super().extra_repr() + f'T={self.T}, '

class DropoutPSN_OR(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan()):
        super().__init__()
        self.psn0 = DropoutPSN(T, surrogate_function)
        self.psn1 = PSN(T, surrogate_function)

    def forward(self, x_seq: torch.Tensor):
        x = self.psn0(x_seq)
        y = self.psn1(x)
        y = x + y - x * y
        return y.view(x_seq.shape)

    def extra_repr(self):
        return super().extra_repr() + f'T={self.T}, '

class PSN_G(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan()):
        super().__init__()
        self.T = T
        self.surrogate_function = surrogate_function
        weight = torch.zeros([T, T])
        bias = torch.zeros([T, 1])
        G = torch.zeros([1, T])
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.G = nn.Parameter(G)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.)
        nn.init.constant_(self.G, 1.)

    def forward(self, x_seq: torch.Tensor):
        weight = self.weight * self.G.sigmoid() 
        h_seq = torch.addmm(self.bias, weight, x_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq.view(x_seq.shape)

    def extra_repr(self):
        return super().extra_repr() + f'T={self.T}, '\
        

class ParameterPSN(nn.Module, base.MultiStepModule):
    def __init__(self, tau: int, T: int, surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan()):
        super().__init__()
        self.T = T
        self.tau = tau
        self.surrogate_function = surrogate_function

        weight_a = torch.zeros([T, tau])
        weight_b = torch.zeros([tau, T])
        bias = torch.zeros([T, 1])

        self.weight_a = nn.Parameter(weight_a)
        self.weight_b = nn.Parameter(weight_b)
        self.bias = nn.Parameter(bias)
        
        lam = math.sqrt(3 / math.sqrt(3 * T * tau))
        nn.init.uniform_(self.weight_a, -lam, lam)
        nn.init.uniform_(self.weight_b, -lam, lam)
        nn.init.constant_(self.bias, -1.)

    def forward(self, x_seq: torch.Tensor):
        weight = self.weight_a @ self.weight_b
        h_seq = torch.addmm(self.bias, weight, x_seq.flatten(1))    
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq.view(x_seq.shape)
    
# class T1PSN(nn.Module, base.SingleModule):
#     def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan()):
#         super().__init__()
#         self.T = T
#         self.surrogate_function = surrogate_function
#         weight1 = torch.zeros([T, 1])
#         bias1 = torch.zeros([1])

#         weight2 = torch.zeros([1, T])
#         bias2 = torch.zeros([1])
        
#         self.weight1 = nn.Parameter(weight1)
#         self.bias1 = nn.Parameter(bias1)
#         self.weight2 = nn.Parameter(weight2)
#         self.bias2 = nn.Parameter(bias2)    

#         nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
#         nn.init.constant_(self.bias1, -1.)
#         nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
#         nn.init.constant_(self.bias2, -1.)

#     def forward(self, x: torch.Tensor):
#         h = torch.addmm(self.bias1, self.weight1, x.flatten(0).unsqueeze(0))
#         spike = self.surrogate_function(h)
#         out = torch.addmm(self.bias2, self.weight2, spike)
#         return out.view(x.shape)

#     def extra_repr(self):
#         return super().extra_repr() + f'T={self.T}, '

@torch.no_grad()
def max_error(x: torch.Tensor, y: torch.Tensor):
    return (x - y).abs().max()

def forward_backward(net: torch.nn.Module, x_seq: torch.Tensor):
    y_seq = net(x_seq)
    y_seq.sum().backward()
    x_seq.grad.zero_()
    functional.reset_net(net)


from spikingjelly.activation_based import neuron,cuda_utils, functional

if __name__ == '__main__':

    T = 8
    N = 64
    C = 32 * 32 * 32
    device = 'cuda:0'

    with torch.cuda.device(device):   
        x_seq = torch.rand([T, N, C], device=device, requires_grad=True, dtype=torch.float32)
        net = ParallelNode(T, 2, surrogate_function=surrogate.Sigmoid()).to(device)
        # net.train()
        y_seq = net(x_seq)
        y_seq.sum().backward()
        x_grad = x_seq.grad.clone()
        x_seq.grad.zero_()
        functional.reset_net(net)

        net.eval()
        y_seq = net(x_seq)
        functional.reset_net(net)

        print(f'max error of y_eval = {max_error(y_seq, y_seq)}')

    # with torch.cuda.device(device):
    #     x_seq = torch.rand([T, N, C], device=device, requires_grad=True, dtype=torch.float32)
    #     SJ_IFNode = IFNode(backend='triton', surrogate_function=surrogate.Sigmoid(), step_mode='m', store_hidden_states=True, )
    #     My_IFNode = IFNode(backend='torch', surrogate_function=surrogate.Sigmoid(), step_mode='m', store_hidden_states=True, )

    #     # SJ_IFNode = LinearNode(a=1.0, b=1.0, learnable=True, backend='triton', surrogate_function=surrogate.Sigmoid(), step_mode='m', store_hidden_states=True, )
    #     # My_IFNode = LinearNode(a=1.0, b=1.0, learnable=True, backend='torch', surrogate_function=surrogate.Sigmoid(), step_mode='m', store_hidden_states=True, )


    #     SJ_IFNode.train()
    #     y_SJ_IFNode = SJ_IFNode(x_seq)
    #     y_SJ_IFNode.sum().backward()
    #     x_grad_SJ_IFNode = x_seq.grad.clone()
    #     x_seq.grad.zero_()
    #     functional.reset_net(SJ_IFNode)

    #     My_IFNode.train()
    #     y_My_IFNode = My_IFNode(x_seq)
    #     y_My_IFNode.sum().backward()
    #     x_grad_My_IFNode = x_seq.grad.clone()
    #     x_seq.grad.zero_()
    #     functional.reset_net(My_IFNode)

    #     print(f'max error of y = {max_error(y_My_IFNode, y_SJ_IFNode)}')
    #     print(f'max error of x_seq.grad = {max_error(x_grad_My_IFNode, x_grad_SJ_IFNode)}')
    #     # print(f'SJ_IFNode.a.grad = {SJ_IFNode.a.grad}, SJ_IFNode.b.grad = {SJ_IFNode.b.grad}')
    #     # print(f'My_IFNode.a.grad = {My_IFNode.a.grad}, My_IFNode.b.grad = {My_IFNode.b.grad}')

    #     SJ_IFNode.eval()
    #     y_SJ_IFNode = SJ_IFNode(x_seq)
    #     functional.reset_net(SJ_IFNode)

    #     My_IFNode.eval()
    #     y_My_IFNode = My_IFNode(x_seq)
    #     functional.reset_net(My_IFNode)

    #     print(f'max error of y_eval = {max_error(y_My_IFNode, y_SJ_IFNode)}')
        
# if __name__ == '__main__':
#     N = 64
#     device = 'cuda:0'

#     repeats = 32

#     for surrogate_function in surrogate._has_cuda_:
#         print(f'surrogate_function = {surrogate_function}')
#         net_triton = IFNode(backend='triton', surrogate_function=surrogate_function(), step_mode='m', detach_reset=True)
#         net_cupy = neuron.IFNode(backend='cupy', surrogate_function=surrogate_function(), step_mode='m', detach_reset=True)

#         for dtype in [torch.half, torch.float]:
#             for C in [32 * 32, 32 * 32  * 32, 32 * 32 * 32 * 4, 32 * 32 * 32 * 8]:
#                 print('N * C = ', N * C)
#                 for T in [2, 4, 8, 16, 32]:
#                     x_seq = torch.rand([T, N, C], device=device, requires_grad=True, dtype=torch.float16)

#                     t_cupy = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_cupy, x_seq)
#                     t_triton = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_triton, x_seq)

#                     print(f'dtype={dtype}, T={T},'.ljust(30), f'net_cupy / t_triton = {round(t_cupy / t_triton, 2)}')
