import torch

""" Gradients computations """

def gradients(outputs, inputs, **kwargs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

def gradients_unused(outputs, inputs, **kwargs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, allow_unused=True)[0]

def gradients_NoGraph(outputs, inputs, **kwargs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=False)[0]

def gradients_clipped(outputs, inputs, **kwargs):
    return [torch.clamp(torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0], -1, 1)]

def gradients_normalized(outputs, inputs, **kwargs):
    gradients = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    return torch.nn.functional.normalize(gradients, p=2, dim=1)
    # return gradients / torch.norm(gradients, p=float('inf'), dim=1).unsqueeze(1).repeat(1, gradients.shape[1], 1)

def gradients_normalizedAdam(outputs, inputs, **kwargs):
    gradients = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    return gradients / torch.sqrt(gradients**2)
    # return gradients / torch.norm(gradients, p=float('inf'), dim=1).unsqueeze(1).repeat(1, gradients.shape[1], 1)

def gradients_adam(outputs, inputs, mtm1, beta1):
    gradients = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    gr = beta1 * mtm1 + (1-beta1) * gradients
    vt = gradients**2
    return gr / torch.sqrt(vt + 1e-8)


def finite_diff(outputs, inputs): # TODO
    """
    outputs = B, X, C
    input = B, X, D

    return = B, X, C, D
    """
    dim = inputs.shape[-1]
    channels = outputs.shape[-1]

    num = torch.diff(outputs, dim=1) # B, X-1, C
    denom = torch.diff(inputs, dim=1) # B, X-1, D

    num = num.unsqueeze(-1).repeat(1, 1, 1, dim)
    denom = denom.unsqueeze(-2).repeat(1, 1, channels, 1)
    
    return num / denom

def compute_jac(): # TODO
    pass

def compute_hess(): # TODO
    pass

if __name__=='__main__':
    a = torch.randn((10, 100, 1))
    b = torch.randn((10, 100, 2))
    finite_diff(a, b)

