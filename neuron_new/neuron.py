from abc import abstractmethod
import torch
import torch.nn as nn
from typing import Callable
import spikingjelly.activation_based.surrogate as surrogate
from spikingjelly.activation_based import base 
import neuron_backend

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
                 step_mode='s', backend='torch', store_v_seq: bool = False):
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
        
        self.store_v_seq = store_v_seq

        # used in lava_exchange
        self.lava_s_cale = 1 << 6
    
    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError
    
    def neuronal_fire(self, h):
        return neuronal_fire(self.surrogate_function, h - self.v_threshold)

    def neuronal_reset(self, h: torch.Tensor, spike: torch.Tensor):
        return neuronal_reset(h, spike, self.detach_reset, self.v_reset, self.v_threshold)
     
    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        h = self.neuronal_charge(x)
        spike = self.neuronal_fire(h)
        self.v = self.neuronal_reset(h, spike)

        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
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

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)

class IFNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, 
                 step_mode='s', backend='torch', store_v_seq: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

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
    
    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
        h = v + x
        spike = (h >= v_threshold).to(x) 
        v = h - spike * v_threshold if v_reset is None else v_reset * spike + (1. - spike) * h
        return spike, v
        
    @staticmethod
    def jit_eval_multi_step_forward(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float, store_v_seq: bool):
        @torch.jit.script
        def multi_step_forward_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
            spike_seq = []
            v_seq = []
            for x in x_seq:
                h = v + x
                spike = (h >= v_threshold).to(x) 
                v = h - spike * v_threshold if v_reset is None else v_reset * spike + (1. - spike) * h
                spike_seq.append(spike)
                v_seq.append(v)
            return torch.stack(spike_seq), v, torch.stack(v_seq)
        
        @torch.jit.script
        def multi_step_forward(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
            spike_seq = []
            for x in x_seq:
                h = v + x
                spike = (h >= v_threshold).to(x) 
                v = h - spike * v_threshold if v_reset is None else v_reset * spike + (1. - spike) * h
                spike_seq.append(spike)
            return torch.stack(spike_seq), v
        
        if store_v_seq:
            return multi_step_forward_with_v_seq(x_seq, v, v_threshold, v_reset)
        else:
            return multi_step_forward(x_seq, v, v_threshold, v_reset)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().multi_step_forward(x_seq)
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
            if self.store_v_seq:
                spike_seq, self.v, self.v_seq = IFNode.jit_eval_multi_step_forward(x_seq, self.v, self.v_threshold, self.v_reset, self.store_v_seq)
            else:
                spike_seq, self.v = IFNode.jit_eval_multi_step_forward(x_seq, self.v, self.v_threshold, self.v_reset, self.store_v_seq)

            return spike_seq   
        
    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().single_step_forward(x)
            elif self.backend == 'triton':
                forward_kernel, backward_kernel =  neuron_backend.IFNode_single_step_forward_kernel, neuron_backend.IFNode_single_step_backward_kernel

                self.v_float_to_tensor(x)

                spike, v = neuron_backend.IFNodeSingleStepATGF.apply(
                                                x.flatten(0),
                                                self.v.flatten(0),
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
        else:
            self.v_float_to_tensor(x)
            spike, self.v = IFNode.jit_eval_single_step_forward(x, self.v, self.v_threshold, self.v_reset)
            return spike

class LinearNode(BaseNode):
    def __init__(self,a: float=1.0, b:float=1.0, learnable: bool = False, v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode = 'm',
                 backend='torch', store_v_seq: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
      
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

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float,
                                     a: float, b: float):
        h = a * v + b * x
        spike = (h >= v_threshold).to(x)
        v = h - spike * v_threshold if v_reset is None else v_reset * spike + (1. - spike) * h
        return spike, v

    @staticmethod
    def jit_eval_multi_step_forward(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float,
                                    a: float, b: float, store_v_seq: bool):
        @torch.jit.script
        def multi_step_forward_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float,
                                            a: float, b: float):
                spike_seq = []
                v_seq = []
                for x in x_seq:
                    h = a * v + b * x
                    spike = (h >= v_threshold).to(x)
                    v = h - spike * v_threshold if v_reset is None else v_reset * spike + (1. - spike) * h
                    spike_seq.append(spike)
                    v_seq.append(v)
                return torch.stack(spike_seq), v, torch.stack(v_seq)
        
        @torch.jit.script
        def multi_step_forward(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float,
                                a: float, b: float):
                spike_seq = []
                for x in x_seq:
                    h = a * v + b * x
                    spike = (h >= v_threshold).to(x)
                    v = h - spike * v_threshold if v_reset is None else v_reset * spike + (1. - spike) * h
                    spike_seq.append(spike)
                return torch.stack(spike_seq), v
        
        if store_v_seq:
            return multi_step_forward_with_v_seq(x_seq, v, v_threshold, v_reset, a, b)
        else:
            return multi_step_forward(x_seq, v, v_threshold, v_reset, a, b)

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().single_step_forward(x)
            elif self.backend == 'triton':
                forward_kernel, backward_kernel = neuron_backend.LinearNode_single_step_forward_kernel, neuron_backend.LinearNode_single_step_backward_kernel

                self.v_float_to_tensor(x)
     
                spike, v = neuron_backend.LinearNodeSingleStepATGF.apply(
                                                x.flatten(0),
                                                self.v.flatten(0),
                                                self.a.to(x.device), self.b.to(x.device), self.learnable,
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
        else:
            self.v_float_to_tensor(x)
            spike, self.v = LinearNode.jit_eval_single_step_forward(x, self.v, self.v_threshold, self.v_reset,
                                                         self.a, self.b)
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().multi_step_forward(x_seq)
            elif self.backend == 'triton':
                forward_kernel, backward_kernel = neuron_backend.LinearNode_multi_step_forward_kernel, neuron_backend.LinearNode_multi_step_backward_kernel

                self.v_float_to_tensor(x_seq[0])
                spike_seq, v_seq = neuron_backend.LinearNodeMultiStepATGF.apply(
                                                    x_seq.flatten(1), 
                                                    self.v.flatten(0),
                                                    self.a.to(x_seq.device), self.b.to(x_seq.device), self.learnable,
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

            if self.store_v_seq:
                spike_seq, self.v, self.v_seq = LinearNode.jit_eval_multi_step_forward(x_seq, self.v, self.v_threshold, self.v_reset,
                                                                            self.a, self.b, self.store_v_seq)
            else:
                spike_seq, self.v = LinearNode.jit_eval_multi_step_forward(x_seq, self.v, self.v_threshold, self.v_reset,
                                                                self.a, self.b, self.store_v_seq)
                
            return spike_seq

@torch.no_grad()
def max_error(x: torch.Tensor, y: torch.Tensor):
    return (x - y).abs().max()

if __name__ == '__main__':
    from spikingjelly.activation_based import neuron,cuda_utils, functional

    T = 8
    N = 64
    C = 32 * 32 * 32
    device = 'cuda:2'
    

    with torch.cuda.device(device):
        # for surrogate in surrogate._has_cuda_:
            # print(surrogate)
        x_seq = torch.rand([T, N, C], device=device, requires_grad=True, dtype=torch.float16)
        SJ_IFNode = LinearNode(a=1.0, b=1.0, learnable=True, backend='torch', surrogate_function=surrogate.Sigmoid(), step_mode='s', store_v_seq=True, )
        My_IFNode = IFNode(backend='triton', surrogate_function=surrogate.Sigmoid(), step_mode='s', store_v_seq=True, )


        SJ_IFNode.train()
        y_SJ_IFNode = SJ_IFNode(x_seq)
        y_SJ_IFNode.sum().backward()
        x_grad_SJ_IFNode = x_seq.grad.clone()
        x_seq.grad.zero_()
        functional.reset_net(SJ_IFNode)

        My_IFNode.train()
        y_My_IFNode = My_IFNode(x_seq)
        y_My_IFNode.sum().backward()
        x_grad_My_IFNode = x_seq.grad.clone()
        x_seq.grad.zero_()
        functional.reset_net(My_IFNode)

        print(f'max error of y = {max_error(y_My_IFNode, y_SJ_IFNode)}')
        print(f'max error of x_seq.grad = {max_error(x_grad_My_IFNode, x_grad_SJ_IFNode)}')

        SJ_IFNode.eval()
        y_SJ_IFNode = SJ_IFNode(x_seq)
        functional.reset_net(SJ_IFNode)

        My_IFNode.eval()
        y_My_IFNode = My_IFNode(x_seq)
        functional.reset_net(My_IFNode)

        print(f'max error of y_eval = {max_error(y_My_IFNode, y_SJ_IFNode)}')

