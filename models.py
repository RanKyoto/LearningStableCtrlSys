from typing import Dict, List, Tuple, Type, Union, Callable
import torch as th
import torch.nn as nn
from networks import LyapunovFunction,create_mlp

class Stable_Dynamics(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 net_arch: List[int] = [32,32],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 device: Union[th.device, str] = "auto"
                 ):
        """
        Initializes a neural network with specified architecture, activation function,
        and device configuration.

        Parameters:
            state_dim (int): Dimension of the input state space, i.e., the number of input features.
            action_dim (int): Dimension of the action space, i.e., the number of output units.
            net_arch (List[int], optional): List specifying the number of units in each hidden layer
                of the neural network. Default is [32, 32], indicating two hidden layers with 32 units each.
            activation_fn (Type[nn.Module], optional): Activation function to be used between layers.
                Default is nn.ReLU. Should be a subclass of nn.Module.
            device (Union[th.device, str], optional): Specifies the device for computation (e.g., 'cpu',
                'cuda', or 'auto' to select automatically based on availability). Default is "auto".
        """
        super().__init__()
        """stable dynamics init"""
        self.device = device
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.x_stable = th.zeros([1,state_dim],dtype=th.float32, device=device).to(device)
        self.f_hat = create_mlp(state_dim,state_dim,net_arch).to(device)
        self.lyapunov_function = LyapunovFunction(state_dim,
                                                  net_arch,
                                                  activation_fn,
                                                  device=device)
        self.alpha_hat = create_mlp(state_dim,action_dim,net_arch).to(device)
        self.g = nn.ModuleList([create_mlp(state_dim,state_dim,net_arch).to(device) for _ in range(action_dim)])
    
    def forward(self, x:th.Tensor):
        f0 = (self.f_hat(x) - self.f_hat(self.x_stable)).unsqueeze(dim=2)
        alpha = (self.alpha_hat(x) - self.alpha_hat(self.x_stable)).unsqueeze(dim=2)
        g = th.stack([gi(x).reshape(-1, self.state_dim,1) for gi in self.g],dim=2).squeeze(dim=3)
        
        V = self.lyapunov_function.forward(x)
        V.sum().backward()
        grad_V = x.grad.unsqueeze(dim=2)
        
        W = 0.1 * th.norm(x.unsqueeze(dim=2),2,dim=1,keepdim=True) ** 2

        criterion = grad_V.transpose(1,2) @ (f0 + g @ alpha) + W
        fs = -  criterion / (th.norm(grad_V,2,dim=1,keepdim=True) ** 2 + 1e-6) * grad_V
        
        mask = criterion <= 0
        f = th.where(mask, f0, f0 + fs)

        del x
        return f, g, alpha, V
    
class Safty_Dynamics(nn.Module):
    def __init__(self,
                state_dim: int, 
                action_dim: int,
                eta:Callable,
                c4:float, 
                net_arch: List[int] = [32, 32],
                activation_fn: Type[nn.Module] = nn.ReLU, 
                device: Union[th.device, str] = "auto"
                ):
         
        super().__init__()
        """stable dynamics init"""
        self.device = device
        self.state_dim = state_dim 
        self.action_dim = action_dim

        self.eta = eta
        self.c4 = c4
    
        self.f_hat = create_mlp(state_dim,state_dim,net_arch).to(device)
        self.alpha = create_mlp(state_dim,action_dim,net_arch).to(device)
        self.g = nn.ModuleList([create_mlp(state_dim,state_dim,net_arch).to(device) for _ in range(action_dim)])

    def forward(self, x:th.Tensor):
        f_hat = self.f_hat(x).unsqueeze(dim=2)
        alpha = self.alpha(x).unsqueeze(dim=2)
        g = th.stack([gi(x).reshape(-1, self.state_dim,1) for gi in self.g],dim=2).squeeze(dim=3)
        
        eta = self.eta(x).reshape(-1,1,1)
        eta.sum().backward(retain_graph=True)
        grad_eta = x.grad.unsqueeze(dim=2)

        criterion = grad_eta.transpose(1,2) @ (f_hat + g @ alpha) + self.c4 * eta
        f_eta = -  criterion / (th.norm(grad_eta,2,dim=1,keepdim=True) ** 2 + 1e-6) * grad_eta
        
        mask = criterion <= 0
        f = th.where(mask, f_hat, f_hat + f_eta)

        del x
        return f, g, alpha

class L2_Dynamics(Stable_Dynamics):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 net_arch: List[int] = [32,32],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 device: Union[th.device, str] = "auto"
                 ):
        super().__init__(state_dim, action_dim, net_arch, activation_fn, device)
       
        
    def forward(self, x:th.Tensor):
        f0 = (self.f_hat(x) - self.f_hat(self.x_stable)).unsqueeze(dim=2)
        alpha = (self.alpha_hat(x) - self.alpha_hat(self.x_stable)).unsqueeze(dim=2)
        g = th.stack([gi(x).reshape(-1, self.state_dim,1) for gi in self.g],dim=2).squeeze(dim=3)
        
        V = self.lyapunov_function.forward(x)
        V.sum().backward()
        grad_V = x.grad.unsqueeze(dim=2)
        
        # please modify W for different dynamics, this is only for van der Pol
        W = (x[:,0] ** 2).reshape(-1,1,1) + grad_V[:,1,:].reshape(-1,1,1)

        criterion = grad_V.transpose(1,2) @ (f0 + g @ alpha) + W
        fs = -  criterion / (th.norm(grad_V,2,dim=1,keepdim=True) ** 2 + 1e-6) * grad_V
        
        mask = criterion <= 0
        f = th.where(mask, f0, f0 + fs)

        del x
        return f, g, alpha

class HJI_Dynamics(Stable_Dynamics):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 net_arch: List[int] = [32,32],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 device: Union[th.device, str] = "auto"
                 ):
        super().__init__(state_dim, action_dim, net_arch, activation_fn, device)
       
        
    def forward(self, x:th.Tensor):
        f0 = (self.f_hat(x) - self.f_hat(self.x_stable)).unsqueeze(dim=2)
        alpha = (self.alpha_hat(x) - self.alpha_hat(self.x_stable)).unsqueeze(dim=2)
        g = th.stack([gi(x).reshape(-1, self.state_dim,1) for gi in self.g],dim=2).squeeze(dim=3)
        
        V = self.lyapunov_function.forward(x)
        V.sum().backward()
        grad_V = x.grad.unsqueeze(dim=2)
        
        # please modify W for different dynamics, this is only for van der Pol
        W = (x[:,0] ** 2).reshape(-1,1,1) + grad_V[:,1,:].reshape(-1,1,1)

        criterion = grad_V.transpose(1,2) @ (f0 + g @ alpha) + W
        fs = -  criterion / (th.norm(grad_V,2,dim=1,keepdim=True) ** 2 + 1e-6) * grad_V
        
        mask = criterion <= 0
        f = th.where(mask, f0, f0 + fs)

        del x
        return f, g, alpha