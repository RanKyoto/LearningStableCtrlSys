from typing import Dict, List, Tuple, Type, Union, Any, Optional
import torch as th
import torch.nn as nn
from utils import get_device, SmoothReLU
from dynamics import CartPole,Pendulum,Dynamics,VanDerPol
from models import Stable_Dynamics,Safty_Dynamics,L2_Dynamics
from torch.utils.data import DataLoader,random_split
import numpy as np
from plots.plot_model import plot_pendulum,plot_cartpole,plot_vanderpol

class MLAgent(nn.Module):
    def __init__(self,
                 name:str,
                 dynamics:Type[Dynamics],
                 model_class:Type[Stable_Dynamics],
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 batch_size:int = 256,
                 data_size:int =31,
                 lr:float = 0.01,
                 activation_fn: Type[nn.Module] = SmoothReLU,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.AdamW,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 device: Union[th.device, str] = "auto") -> None:
        """
        Initializes the model with specified dynamics, model type, training parameters,
        and device configuration, preparing it for training or evaluation.

        Parameters:
            name (str): A unique identifier or name for this instance.
            dynamics (Type[Dynamics]): Class type for the dynamics system, responsible for
                providing the dataset and initializing environment parameters.
            model (Type[Stable_Dynamics]): Class type of the model to be used. Should be a subclass
                of Stable_Dynamics, which defines the neural network structure and dynamics.
            batch_size (int, optional): Number of samples per batch during training. Default is 256.
            data_size (int, optional): Number of data samples to generate or use from the dataset.
                Default is 31.
            lr (float, optional): Learning rate for the optimizer. Default is 0.01.
            activation_fn (Type[nn.Module], optional): Activation function for the model, defaulting to
                SmoothReLU. Must be a subclass of nn.Module.
            optimizer_class (Type[th.optim.Optimizer], optional): Optimizer class to use for training
                the model. Default is AdamW. Must be a subclass of torch.optim.Optimizer.
            optimizer_kwargs (Optional[Dict[str, Any]], optional): Additional arguments for the optimizer,
                passed as a dictionary. Default is None.
            device (Union[th.device, str], optional): Specifies the device to use (e.g., 'cpu', 'cuda',
                or 'auto' to select automatically based on availability). Default is "auto".
        """
        super().__init__()
        self.name = name
        self.device = get_device(device)

        dynamics = dynamics()
        self.dataset = dynamics.make_dataset(num=data_size,isRandom=False)
        self.size_dataset = len(self.dataset)
        
        if model_kwargs is None:
            model_kwargs = {}
        self.model = model_class(state_dim=dynamics.state_dim,
                                 action_dim=dynamics.action_dim,
                                 activation_fn=activation_fn,
                                 device=self.device,
                                 **model_kwargs)
                
        self.batch_size = batch_size
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = optimizer_class(self.model.parameters(), lr = lr, **optimizer_kwargs)

    def train(self, epoches = 10, isSave = True):
        # [step 1] Prepare dataset
        train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # [step 2] Start training!!
        self.model.train()
        for epoch in range(0, epoches):
            # record MSEloss per epoch
            loss_list = []
            # for batch_index, data in enumerate(train_dataloader):
            for input, output in train_dataloader:
                xu = input.to(self.device)
                dx = output.to(self.device).unsqueeze(dim=2)   
                x = xu[:,0:self.model.state_dim]
                u = xu[:,self.model.state_dim:].unsqueeze(dim=2)   
                x.requires_grad = True
                f, g, alpha = self.model(x)
                  
                self.optimizer.zero_grad()
                loss = nn.functional.mse_loss(f + g @ u, dx)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
            print(epoch, "ave_loss=", np.mean(loss_list))
        if isSave:
            return self.save_model()
        else:
            return None

    def save_model(self):
        scripted_model = th.jit.script(self.model)
        scripted_model.save("./saved_model/{}.zip".format(self.name))
        return scripted_model
    
    def load_model(self):
        return th.jit.load("./saved_model/{}.zip".format(self.name))

if __name__ == "__main__":
    agent = MLAgent(name = "cartpole",
                    dynamics=CartPole,
                    model_class=Stable_Dynamics,
                    batch_size=128,
                    data_size=21,
                    lr=0.01,
                    device="cuda:0")
    #agent.train(10)
    model = agent.load_model()
    plot_cartpole(model)
    # def eta(x): return (x[:, 0] - 1.5)**2 + x[:, 1]**2 - 1
    # agent = MLAgent(name = "safe_vanderpol",
    #                 dynamics=VanDerPol,
    #                 model_class=Safty_Dynamics,
    #                 #model_class=Stable_Dynamics,
    #                 batch_size=128,
    #                 data_size=21,
    #                 lr=0.01,
    #                 model_kwargs={"c4":0.1,"eta":eta},
    #                 device="cuda:0")
    # model = agent.train(epoches=5)
    # #model = agent.load_model()
    # plot_vanderpol(model)

    # agent = MLAgent(name = "L2_vanderpol",
    #                 dynamics=VanDerPol,
    #                 model_class=L2_Dynamics,
    #                 batch_size=128,
    #                 data_size=21,
    #                 lr=0.01,
    #                 device="cuda:0")
    # agent.train(10)
  
        
       
    