import torch.nn as nn
import torch as th
import numpy as np
from torch.utils.data import TensorDataset

class Dynamics(nn.Module):
    def __init__(self, state_dim:int, action_dim:int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def make_dataset(self,num:int,isRandom:bool)->TensorDataset:
        return None
    
class VanDerPol(Dynamics):
    def __init__(self, mu: float = 0.3, tau: float = 0.01) -> None:
        """
        Initializes the Van der Pol oscillator dynamics.

        Parameters:
            mu (float): The nonlinearity parameter that controls the strength of
                        the oscillation. Larger values increase oscillatory behavior.
            tau (float): The time step between state updates.
        """
        super().__init__(state_dim=2, action_dim=1)
        self.mu = mu
        self.tau = tau  # time interval for state updates

    def state_dot(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the state (state_dot) based on the current
        state and action input using the Van der Pol equations.

        Parameters:
            state (np.ndarray): The current state of shape (N, 2), where the first
                                column is position `x`, and the second is velocity `x_dot`.
            action (np.ndarray): Control input, though it typically does not affect
                                 Van der Pol dynamics directly (set to 0 for now).

        Returns:
            np.ndarray: The time derivative of the state, state_dot, with shape (N, 2).
        """
        x = state[:, 0]
        x_dot = state[:, 1]
        u = action.flatten()
        # Van der Pol oscillator equations
        x_ddot = - x + self.mu * (1 - x_dot**2) * x_dot  + u
        return self.tau * np.column_stack((x_dot, x_ddot))

    def plot(self):
        """
        Plots the phase portrait of the Van der Pol oscillator.
        """
        import matplotlib.pyplot as plt

        # Define the ranges for `x` and `x_dot` and create a grid
        x = np.linspace(-3, 3, 200)
        x_dot = np.linspace(-3, 3, 200)
        x, x_dot = np.meshgrid(x, x_dot)
        x = x.flatten()
        x_dot = x_dot.flatten()

        # Generate state derivatives for each point in the grid
        state = np.column_stack((x, x_dot))
        action = np.zeros_like(x)  # Action is zero for Van der Pol dynamics
        state_dot = self.state_dot(state=state, action=action)

        # Compute the speed of state change for color mapping
        dx, dx_dot = state_dot[:, 0], state_dot[:, 1]
        speed = np.sqrt(dx**2 + dx_dot**2)

        # Plot the phase portrait using a streamplot
        plt.figure(figsize=(8, 8))
        plt.streamplot(
            x.reshape(200, 200), x_dot.reshape(200, 200),
            dx.reshape(200, 200), dx_dot.reshape(200, 200),
            color=speed.reshape(200, 200), cmap='autumn', arrowsize=5, linewidth=2.5
        )

        plt.xlabel(r"$x$", fontsize=18)
        plt.ylabel(r"$\dot{x}$", fontsize=18)
        plt.title("Van der Pol Oscillator Phase Portrait", fontsize=20)
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        plt.show()    

    def make_dataset(self, num=11, isRandom=True) -> TensorDataset:
        """
        Creates a dataset of state-action pairs and their derivatives for the Van der Pol oscillator.

        Parameters:
            num (int): Number of samples to generate along each dimension.
            isRandom (bool): If True, sample randomly within the range; if False, sample linearly.

        Returns:
            TensorDataset: A dataset containing input states and their derivatives.
        """
        # Define the range for the states `x` and `x_dot`
        x_range = 3.0
        u_range = 1
        # Randomly or linearly sample initial conditions for `x` and `x_dot`
        if isRandom:
            x = np.random.uniform(-x_range, x_range, num)
            x_dot = np.random.uniform(-x_range, x_range, num)
            u = np.random.uniform(0, u_range, 2)
        else:
            x = np.linspace(-x_range, x_range, num)
            x_dot = np.linspace(-x_range, x_range, num)
            u = np.linspace(0, u_range, 2)


        # Generate a grid for all combinations of `x` and `x_dot`
        x, x_dot, u = np.meshgrid(x, x_dot, u)
        state = np.array([x, x_dot]).reshape(2, -1).T
        action = np.array([u]).reshape(1, -1).T
        # Convert state and action to PyTorch tensors
        input_data = th.from_numpy(np.hstack((state, action)).astype(np.float32)).clone()

        # Compute state derivatives
        output_data = self.state_dot(state, action)
        output_data = th.from_numpy(output_data.astype(np.float32)).clone()

        # Return the dataset
        dataset = TensorDataset(input_data, output_data)
        return dataset

class CartPole(Dynamics):  
    def __init__(self) -> None:
        super().__init__(state_dim=4, action_dim=1)     
        """
        OpenAI gymnasium CartPole-v1
        
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        """
        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.x_max = 1.0
        self.x_dot_max = 1.0
        self.theta_max = np.pi/6
        self.theta_dot_max = 1.0


    def state_dot(self, state:np.ndarray, action:np.ndarray):
        x = state[:, 0]
        x_dot = state[:, 1]
        theta = state[:, 2]
        theta_dot = state[:, 3]

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            action + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        return self.tau * np.column_stack((x_dot, xacc, theta_dot, thetaacc))
    
    def plot(self):
        import matplotlib.pyplot as plt
        
        theta = np.linspace(-self.theta_max, self.theta_max, 200)
        theta_dot = np.linspace(-self.theta_dot_max, self.theta_dot_max, 200)
       
        theta, theta_dot = np.meshgrid(theta,theta_dot)
        theta = theta.flatten()
        theta_dot = theta_dot.flatten()
        x, x_dot,  u= np.zeros_like(theta), np.ones_like(theta)*0, np.ones_like(theta)*0
        state = np.column_stack((x, x_dot, theta, theta_dot))
       
        state_dot = self.state_dot(state=state,action=u)
        dtheta,dtheta_dot = state_dot[:,2],state_dot[:,3]
        speed = np.sqrt(dtheta**2 + dtheta_dot**2)
        
        plt.figure(figsize=(8,8))
        plt.streamplot(theta.reshape(200,200),
                       theta_dot.reshape(200,200),
                       dtheta.reshape(200,200),
                       dtheta_dot.reshape(200,200),
                       color=speed.reshape(200,200),cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

        plt.xlabel(r"$\theta$",fontsize=18)
        plt.ylabel(r"$\dot\theta$",fontsize=18)
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        
        plt.show()

    def make_dataset(self,num=21, isRandom=True)->TensorDataset:
        x = np.linspace(-self.x_max, self.x_max, 5)
        x_dot = np.linspace(-self.x_dot_max, self.x_dot_max, 5)
        if isRandom:
            theta = np.random.uniform(-self.theta_max, self.theta_max, num)
            theta_dot = np.random.uniform(-self.theta_dot_max, self.theta_dot_max, num)    
        else:    
            theta = np.linspace(-self.theta_max, self.theta_max, num)
            theta_dot = np.linspace(-self.theta_dot_max, self.theta_dot_max, num)
        u = np.linspace(-self.force_mag,self.force_mag,num)

        x, x_dot, theta, theta_dot, u = np.meshgrid(x,x_dot,theta,theta_dot,u)
        state_action = np.array([x, x_dot, theta, theta_dot, u]).reshape(5,-1).T
        input_data = th.from_numpy(state_action.astype(np.float32)).clone()
        output_data = self.state_dot(state_action[:,:4],state_action[:,4])
 
        output_data = th.from_numpy(output_data.astype(np.float32)).clone()
        dataset = TensorDataset(input_data, output_data)
        return dataset
    
class Pendulum(Dynamics):
    """OpenAI gymnasium Pendulum-v1"""
    def __init__(self) -> None:
        super().__init__(state_dim=2, action_dim=1)
        self.dt = 0.05
        self.g = 0.98
        self.m = 1.0
        self.l = 1.0

        self.x1_min=-3
        self.x1_max= 3 
        self.x2_min=-3, 
        self.x2_max= 3
        self.u_min = 2
        self.u_max =-2

    def x1_dot(self, x1:np.ndarray, x2:np.ndarray, u:np.ndarray):
        return self.dt*x2 
    
    def x2_dot(self, x1:np.ndarray, x2:np.ndarray, u:np.ndarray):
        return self.dt*(3 * self.g / (2 * self.l) * np.sin(x1) + 3.0 / (self.m * self.l**2) * u)

    def make_dataset(self,num=11)->TensorDataset:    
        tmp_x1 = np.linspace(self.x1_min,self.x1_max,num)
        tmp_x2 = np.linspace(self.x2_min,self.x2_max,num)
        tmp_u = np.linspace(self.u_min,self.u_max,num)
        grid_x1, grid_x2, grid_u = np.meshgrid(tmp_x1,tmp_x2,tmp_u)
        x1 = grid_x1.ravel()
        x2 = grid_x2.ravel()
        u = grid_u.ravel()
        X1 = th.from_numpy(x1.astype(np.float32)).clone()
        X2 = th.from_numpy(x2.astype(np.float32)).clone()
        U = th.from_numpy(u.astype(np.float32)).clone()
        input_data = th.stack([X1, X2, U], 1)
        dx1 = self.x1_dot(x1,x2,u)
        dx2 = self.x2_dot(x1,x2,u)    
        DX1 = th.from_numpy(dx1.astype(np.float32)).clone()
        DX2 = th.from_numpy(dx2.astype(np.float32)).clone()
        output_data = th.stack([DX1,DX2],1)
        dataset = TensorDataset(input_data, output_data)
        return dataset
    
    def plot(self):
        import matplotlib.pyplot as plt
        x1 = np.linspace(self.x1_min, self.x1_max, 200)
        x2 = np.linspace(self.x2_min, self.x2_max, 200)
       
        x, y = np.meshgrid(x1,x2)
        u = 0
       
        dx1 = self.x1_dot(x,y,u)
        dx2 = self.x2_dot(x,y,u)
        speed = np.sqrt(dx1**2 + dx2**2)
        
        plt.figure(figsize=(8,8))
        plt.streamplot(x,y,dx1,dx2,color=speed,cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

        plt.xlabel("x_1",fontsize=18)
        plt.ylabel("x_2",fontsize=18)
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        
        plt.show()


    
        
    
if __name__ == "__main__":
    # p = Pendulum()
    # p.plot()
    # p.make_dataset()
    cp = VanDerPol()
    cp.plot()
    #cp.make_dataset()




