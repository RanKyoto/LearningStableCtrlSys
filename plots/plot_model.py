import torch as th
import numpy as np
import sys
sys.path.append("./")
import dynamics as dyn


def plot_cartpole(model:th.ScriptModule):
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True) # use latex
    label_size = 20
    theta_max = np.pi/6
    theta_dot_max = 1.0
    theta = np.linspace(-theta_max, theta_max, 200)
    theta_dot = np.linspace(-theta_dot_max, theta_dot_max, 200)
    
    theta_tmp, theta_dot_tmp = np.meshgrid(theta,theta_dot)
    theta = theta_tmp.flatten()
    theta_dot = theta_dot_tmp.flatten()
    x, x_dot= np.zeros_like(theta), np.ones_like(theta)*0
    state = np.column_stack((x, x_dot, theta, theta_dot))
    state_tensor= th.tensor(state,dtype=th.float32)
    state_tensor.requires_grad = True

    f,g, alpha = model(state_tensor)

    dx1 = (f[:,2,0].reshape(200,200)).cpu().detach().numpy()
    dx2 = (f[:,3,0].reshape(200,200)).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(theta_tmp,
                    theta_dot_tmp,
                    dx1.reshape(200,200),
                    dx2.reshape(200,200),
                    color=speed.reshape(200,200),cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)
    
    #plt.xlabel(r"$x_3$",fontsize=18)
    #plt.ylabel(r"$x_4$",fontsize=18)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig("./figures/cartpole/cartpole_LDNA.png")
    print("LDNA...done.")

    f_close = f + g @ alpha
    dx1 = (f_close[:,2,0].reshape(200,200)).cpu().detach().numpy()
    dx2 = (f_close[:,3,0].reshape(200,200)).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(theta_tmp,theta_dot_tmp,dx1,dx2,color=speed,cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

    #plt.xlabel(r"$x_3$",fontsize=18)
    #plt.ylabel(r"$x_4$",fontsize=18)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig("./figures/cartpole/cartpole_LDLA.png")
    print("LDLA...done.")

    dynamics = dyn.CartPole()
    u= np.zeros_like(theta)
    state_dot = dynamics.state_dot(state=state,action=u)
    dtheta,dtheta_dot = state_dot[:,2],state_dot[:,3]
    speed = np.sqrt(dtheta**2 + dtheta_dot**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(theta.reshape(200,200),
                    theta_dot.reshape(200,200),
                    dtheta.reshape(200,200),
                    dtheta_dot.reshape(200,200),
                    color=speed.reshape(200,200),cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

    #plt.xlabel(r"$x_3$",fontsize=18)
    #plt.ylabel(r"$x_4$",fontsize=18)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig("./figures/cartpole/cartpole_RDNA.png")
    print("RDNA...done.")

    u= alpha.cpu().detach().numpy().flatten()
    state_dot = dynamics.state_dot(state=state,action=u)
    dtheta,dtheta_dot = state_dot[:,2],state_dot[:,3]
    speed = np.sqrt(dtheta**2 + dtheta_dot**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(theta.reshape(200,200),
                    theta_dot.reshape(200,200),
                    dtheta.reshape(200,200),
                    dtheta_dot.reshape(200,200),
                    color=speed.reshape(200,200),cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

    #plt.xlabel(r"$x_3$",fontsize=18)
    #plt.ylabel(r"$x_4$",fontsize=18)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig("./figures/cartpole/cartpole_RDLA.png")
    print("RDLA...done.")

def plot_vanderpol(model: th.ScriptModule):
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)  # Enable LaTeX rendering
    label_size = 20
    x1_max = 3.0
    x2_max = 3.0
    
    # Set up state space for plotting
    x1 = np.linspace(-x1_max, x1_max, 200)
    x2 = np.linspace(-x2_max, x2_max, 200)
    x1_tmp, x2_tmp = np.meshgrid(x1, x2)
    x1 = x1_tmp.flatten()
    x2 = x2_tmp.flatten()
    state = np.column_stack((x1, x2))
    state_tensor = th.tensor(state, dtype=th.float32)
    state_tensor.requires_grad = True

    # Obtain the dynamics (f) and control term (g, alpha) from the model
    f, g, alpha = model(state_tensor)

    # Plot the open-loop dynamics (without control)
    dx1 = f[:, 0].reshape(200, 200).cpu().detach().numpy()
    dx2 = f[:, 1].reshape(200, 200).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x1_tmp, x2_tmp, dx1, dx2, color=speed, cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig("./figures/vanderpol/vanderpol_LDNA.png")
    print("Open-loop dynamics (LDNA)...done.")

    # Plot the closed-loop dynamics (with control)
    f_close = f + g @ alpha
    dx1 = f_close[:, 0].reshape(200, 200).cpu().detach().numpy()
    dx2 = f_close[:, 1].reshape(200, 200).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x1_tmp, x2_tmp, dx1, dx2, color=speed, cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig("./figures/vanderpol/vanderpol_LDLA.png")
    print("Closed-loop dynamics (LDLA)...done.")

    # Plot real dynamics without control using `VanDerPol` class directly
    dynamics = dyn.VanDerPol()
    u = np.zeros_like(x1)
    state_dot = dynamics.state_dot(state=state, action=u)
    dx1_real, dx2_real = state_dot[:, 0], state_dot[:, 1]
    speed = np.sqrt(dx1_real**2 + dx2_real**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x1_tmp, x2_tmp, dx1_real.reshape(200, 200), dx2_real.reshape(200, 200), color=speed.reshape(200, 200),
                   cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig("./figures/vanderpol/vanderpol_RDNA.png")
    print("Real dynamics without control (RDNA)...done.")

    # Plot real dynamics with control
    u = alpha.cpu().detach().numpy().flatten()
    state_dot = dynamics.state_dot(state=state, action=u)
    dx1_real, dx2_real = state_dot[:, 0], state_dot[:, 1]
    speed = np.sqrt(dx1_real**2 + dx2_real**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x1_tmp, x2_tmp, dx1_real.reshape(200, 200), dx2_real.reshape(200, 200), color=speed.reshape(200, 200),
                   cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig("./figures/vanderpol/vanderpol_RDLA.png")
    print("Real dynamics with control (RDLA)...done.")

    

def plot_pendulum(model:th.ScriptModule):
    import matplotlib.pyplot as plt
    x1 = np.linspace(-3, 3, 200)
    x2 = np.linspace(-3, 3, 200)
    
    x, y = np.meshgrid(x1,x2)
    state = th.tensor(np.vstack((x.flatten(),y.flatten())).T,
                        dtype=th.float32)
    state.requires_grad = True
    f, g, alpha = model(state)
 
    dx1 = (f[:,0,0].reshape(200,200)).cpu().detach().numpy()
    dx2 = (f[:,1,0].reshape(200,200)).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(x,y,dx1,dx2,color=speed,cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

    plt.xlabel("$x_1$",fontsize=18)
    plt.ylabel("$x_2$",fontsize=18)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig("learned_dynamics.png")

    f_close = f + g @ alpha
    dx1 = (f_close[:,0,0].reshape(200,200)).cpu().detach().numpy()
    dx2 = (f_close[:,1,0].reshape(200,200)).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(x,y,dx1,dx2,color=speed,cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

    plt.xlabel("$x_1$",fontsize=18)
    plt.ylabel("$x_2$",fontsize=18)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig("learned_fb_dynamics.png")