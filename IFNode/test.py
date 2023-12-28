import torch

class Test(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input * 2
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * 2
        return grad_input
    

x = torch.tensor([1.0], requires_grad=True)
c = Test.apply(x)
c.backward()

print("Input: ", x.data)
print("Output: ", c.data)
print("Gradient: ", x.grad.data)
