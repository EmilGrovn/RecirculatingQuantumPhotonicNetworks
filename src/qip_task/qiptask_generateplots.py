#qiptask_generate_plots.py

#----------------------------------
#PLOTTING
#--------------------------------
def generate_plots(inner_product_list,parameter_list,spec,params_amp_dict):
    plot_inner_product(inner_product_list)      #to verify convergence
    params_train_dict=parameter_list[-1]
    plot_ham_and_physical_rates(spec,params_train_dict,params_amp_dict)
    #plot_timevolve_logbasis_one(parameter_list[-1])
    plot_coeffmatrix(spec,params_train_dict,params_amp_dict)
    plot_unitarymatrix(spec,params_train_dict,params_amp_dict)    

### inner product plot
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
def plot_inner_product(inner_products_list):
    """
    Plot the absolute value and angle (phase) of the inner products versus epochs.
    inner_products is expected to be an iterable of complex numbers.
    """
    # Convert inner_products to numpy array if needed
    inner_products=jnp.stack(inner_products_list,axis=0)
    epochs=np.arange(1,inner_products.shape[0]+1)
    inner_products=jnp.squeeze(inner_products)
    abs_vals = jnp.abs(inner_products)
    angles = jnp.angle(inner_products)
    #print(angles.shape,angles[:,0].shape)
    #angles_rel=angles-angles[:,0][:,None]
    #print(angles.shape,angles_rel)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, abs_vals, linestyle='--', label='|Inner Product|')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('Inner Product Magnitude vs Epoch')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, angles/jnp.pi, linestyle='--', label='Angle (radians)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Angle (rad)')
    ax2.set_title('Inner Product Angle vs Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    #plt.show()
    wandb.log({"Inner products plot": wandb.Image(fig)})    

#plot physical rates
from rqpnn.rqpnn_parameterization import generate_ham_properties_from_params
from qip_task.qiptask_costfunc import compute_physical_rates
def plot_ham_and_physical_rates(spec,params_train_dict,params_amp_dict,opt_running=True):
    params_ham_dict=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    rate_tuple=compute_physical_rates(spec,params_train_dict,params_amp_dict)
    
    #physical rates
    tau_rates=rate_tuple[0]
    delta_cav_physical=rate_tuple[1]
    kappa_physical=rate_tuple[2]
    tphys=jnp.zeros(shape=(len(tau_rates)+1))
    for i in range(len(tau_rates)):
        tphys=tphys.at[i+1].set(tphys[i]+tau_rates[i])
    #tphys=jnp.pad(tau_rates,1)[:-1]
    fig, axs=pwc_plot_list_subplots(spec,tphys,delta_cav_physical,rate_type="delta_cav",physical=True)
    if opt_running==True:
        wandb.log({"delta_cav phys time plot": wandb.Image(fig)})
    else:
        plt.show()
    fig, axs=pwc_plot_list_subplots(spec,tphys,kappa_physical,rate_type="kappa",physical=True)
    if opt_running==True:
        wandb.log({"kappa phys time plot": wandb.Image(fig)})
    else:
        plt.show()
    if spec.has_emitters:
        delta_em_physical=rate_tuple[3]
        fig, axs=pwc_plot_list_subplots(spec,tphys,delta_em_physical,rate_type="delta_em",physical=True)
        if opt_running==True:
            wandb.log({"delta_em phys time plot": wandb.Image(fig)})        
        else:
            plt.show()
    #ham parameters
    tau_ham=params_ham_dict['tau']
    delta_cav_ham=params_ham_dict['delta_cav']
    kappa_ham=params_ham_dict['kappa']
    t_ham=jnp.linspace(0,len(tau_ham),len(tau_ham)+1)
    #plot tau
    fig, axs=pwc_plot_list_subplots(spec,t_ham,tau_ham,rate_type="tau",physical=False)
    if opt_running==True:
        wandb.log({"tau ham time plot": wandb.Image(fig)})        
    else:
        plt.show()
    #plot delta_cav
    fig, axs=pwc_plot_list_subplots(spec,t_ham,delta_cav_ham,rate_type="delta_cav",physical=False)
    if opt_running==True:
        wandb.log({"delta_cav ham time plot": wandb.Image(fig)})        
    else:
        plt.show()
    #plot kappa
    fig, axs=pwc_plot_list_subplots(spec,t_ham,kappa_ham,rate_type="kappa",physical=False)
    if opt_running==True:
        wandb.log({"kappa ham time plot": wandb.Image(fig)})        
    else:
        plt.show()
    #plot delta_em
    if spec.has_emitters:
        delta_em_ham=params_ham_dict['delta_em']
        fig, axs=pwc_plot_list_subplots(spec,t_ham,delta_em_ham,rate_type="delta_em",physical=False)
        if opt_running==True:
            wandb.log({"delta_em ham time plot": wandb.Image(fig)})        
        else:
            plt.show()

#helper plotter
def pwc_plot_list_subplots(spec,times, values_list, rate_type="delta_cav",physical=True):
    """
    Plots a piecewise constant function for each array in values_list in its own subplot.
    
    Parameters:
        times (array-like): 1D array of time points.
        values_list (list of array-like): List where each element is an array of values
            corresponding to each time interval.
        rate_label (str): Label for the y-axis.
    """
    # Prepare time values to plot piecewise constant curves:
    # For times: [t0, t1, t2, ...] -> [t0, t1, t1, t2, t2, ...]
    times_plot = np.repeat(times, 2)[1:-1]  # (2n)
    if values_list.ndim==1: #when plotting tau
        values_list=[values_list]
    n_plots = len(values_list)
    fig, axs = plt.subplots(n_plots, 1, figsize=(5, n_plots), sharex=True)
    
    # Ensure axs is always a list (even if n_plots==1)
    if n_plots == 1:
        axs = [axs]
    colors=['b','r','g','m','c','y','k','orange']

    ymin=min(jnp.concatenate(values_list))
    ymax=max(jnp.concatenate(values_list))
    for i, ax in enumerate(axs):
        # Repeat each value to create a step plot
        values = np.repeat(values_list[i], 2)
        # Plot the piecewise constant function
        ax.plot(times_plot, np.real(values),color=colors[i], label=f'cav {i+1}', alpha=0.7)
        ax.fill_between(times_plot, 0, np.real(values),color=colors[i], alpha=0.2)
        if spec.has_emitters==True:
            if physical==True:
                axs[-1].set_xlabel(r'$t*g$')
                if rate_type=="delta_cav":
                    ax.set_ylabel(r"$\delta^{(c)}/g$")
                elif rate_type=="kappa":
                    ax.set_ylabel(r"$\kappa^{(c)}/g$")
                elif rate_type=="delta_em":
                    ax.set_ylabel(r"$\delta^{(e)}/g$")
            elif physical==False:
                axs[-1].set_xlabel(r'$t*g/\tau(t)$')
                if rate_type=="delta_cav":
                    ax.set_ylabel(r"$\tau(t)\delta^{(c)}(t)/g$")
                elif rate_type=="kappa":
                    ax.set_ylabel(r"$\tau(t)\kappa^{(c)}(t)/g$")
                elif rate_type=="delta_em":
                    ax.set_ylabel(r"$\tau(t)\delta^{(e)}(t)/g$")
                elif rate_type=="tau":
                    ax.set_ylabel(r"$\Delta\tau(t)*g$")
        elif spec.has_emitters==False:
            if physical==True:
                axs[-1].set_xlabel(r'$t*\chi_3$')
                if rate_type=="delta_cav":
                    ax.set_ylabel(r"$\delta^{(c)}/\chi_3$")
                elif rate_type=="kappa":
                    ax.set_ylabel(r"$\kappa^{(c)}/\chi_3$")
                elif rate_type=="delta_em":
                    ax.set_ylabel(r"$\delta^{(e)}/\chi_3$")
            elif physical==False:
                axs[-1].set_xlabel(r'$t*\chi_3/\tau(t)$')
                if rate_type=="delta_cav":
                    ax.set_ylabel(r"$\Delta\tau(t)\delta^{(c)}/\chi_3$")
                elif rate_type=="kappa":
                    ax.set_ylabel(r"$\Delta\tau(t)\kappa^{(c)}/\chi_3$")
                elif rate_type=="delta_em":
                    ax.set_ylabel(r"$\Delta\tau(t)\delta^{(e)}/\chi_3$")
                elif rate_type=="tau":
                    ax.set_ylabel(r"$\Delta\tau(t)*g$")
        ax.legend(loc='lower right')
        ax.set_xlim(0, times_plot[-1])
        ax.set_ylim(ymin,ymax)
        ax.grid(True)       
    plt.tight_layout()
    return fig, axs
##
### PLOT MATRICES
import jax
import matplotlib.colors as colors
##plot coefficient matrix
def plot_coeffmatrix(spec,params_train_dict,params_amp_dict, title="Heatmap", xlabel="Column", ylabel="Row",
                        cmap="bwr", colorbar_label="Value", figsize=(8, 6),
                        fmt=".2f",opt_running=True):
    """
    Plots a heatmap of a JAX array with the numerical value of each element shown.
    The colorbar is centered with white corresponding to 0.
    
    Parameters:
    matrix: A jax.numpy array to visualize.
    title (str): Title of the heatmap.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    cmap (str): Colormap to use (default "bwr" for blue-white-red).
    colorbar_label (str): Label for the colorbar.
    figsize (tuple): Figure size in inches.
    fmt (str): String format for annotating each cell.
    """
    params_ham_dict=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    coeff_mat=params_ham_dict["coeff_mat"]
    print("coeff mat",coeff_mat)
    # Convert the JAX array to a NumPy array.
    matrix_np = jax.device_get(coeff_mat.real)
    
    # Create a normalization that centers at 0.
    plot_matrix(matrix_np,"Coefficient matrix","coefficient matrix plot",opt_running=opt_running)

def plot_matrix(matrix_np,title,logname, xlabel="Column", ylabel="Row",
                        cmap="bwr", colorbar_label="Value", figsize=(8, 6),
                        fmt=".2f",opt_running=True):
    amp=max(abs(np.nanmin(matrix_np)),np.nanmax(matrix_np))
    # print("min plot",min(-amp,-0.01))
    # print("max plot",max(amp,0.01))
    norm = colors.TwoSlopeNorm(vmin=min(-amp,-0.01),
                            vcenter=0,
                            vmax=max(amp,0.01))
    fig=plt.figure(figsize=figsize)
    im = plt.imshow(matrix_np, cmap=cmap, norm=norm, aspect='auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Annotate each cell with its numerical value.
    nrows, ncols = matrix_np.shape
    for i in range(nrows):
        for j in range(ncols):
            # For contrast, choose text color based on the background.
            text_color = "white" if norm(matrix_np[i, j]) > 0.5 else "black"
            plt.text(j, i, format(matrix_np[i, j], fmt),
                    ha="center", va="center", color=text_color)
    
    # Create a colorbar and set its label.
    cbar = plt.colorbar(im)
    cbar.set_label(colorbar_label)
    
    #plt.show()
    if opt_running:
        #print("logging matrix")
        wandb.log({logname: wandb.Image(fig)})
##plot unitary matrix
def plot_unitarymatrix(spec,params_train_dict,params_amp_dict, title="Heatmap", xlabel="Column", ylabel="Row",
                        cmap="bwr", colorbar_label="Value", figsize=(8, 6),
                        fmt=".2f",opt_running=True):
    """
    Plots a heatmap of a JAX array with the numerical value of each element shown.
    The colorbar is centered with white corresponding to 0.
    
    Parameters:
    matrix: A jax.numpy array to visualize.
    title (str): Title of the heatmap.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    cmap (str): Colormap to use (default "bwr" for blue-white-red).
    colorbar_label (str): Label for the colorbar.
    figsize (tuple): Figure size in inches.
    fmt (str): String format for annotating each cell.
    """
    params_ham_dict=generate_ham_properties_from_params(spec,params_train_dict,params_amp_dict)
    uni_mat=params_ham_dict["S_in_mat"]

    # Convert the JAX array to a NumPy array.
    matrix_np_real = jax.device_get(uni_mat.real)
    matrix_np_imag = jax.device_get(uni_mat.imag)
    matrix_np_abs = jax.device_get(abs(uni_mat.imag))
    matrix_np_phase = jax.device_get(jnp.angle(uni_mat))
    #matrix_np_symmetric = jax.device_get(uni_mat@uni_mat.T)

    matrix_list=[matrix_np_real,matrix_np_imag,matrix_np_abs,matrix_np_phase]
    title_list=["Real part of unitary","Imag part of unitary","absolute value of unitary elements","Phase of unitary elements"]
    log_list=["unitary matrix real part plot","unitary matrix imag part plot","unitary matrix abs plot","unitary matrix phase plot"]
    for i in range(len(matrix_list)):
        plot_matrix(matrix_list[i],title_list[i],log_list[i])