import numpy as np
import matplotlib.pyplot as plt

def plot_diode_fit(V_data, I_data, model, fitted_params, filename=None, temps=None):
    """
    Generates a comparison of the I-V data and the I-V curve from the fitted parameters

    Args:
        V_data (scalar/numpy array): voltage from I-V data
        I_data (scalar/numpy array): current from I-V data
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        filename (optional): filename for saving the error plot locally, defaults to None
        temps (optional): array of temperatures for multi-temp diode curves, defaults to None
    """ 
    if temps is None:
        plt.semilogy(V_data, I_data, label='Original data')
        I_fit = model.compute_current(V_data, fitted_params)
        plt.semilogy(V_data, I_fit, label='Fitted data')
        plt.legend()
        plt.xlabel("Voltage [V]")
        plt.ylabel("Current [A]")
    
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        for i, T in enumerate(temps):
            local_params = fitted_params.copy()
            if 'Eg' in fitted_params:
                local_params['I_s'] = model.compute_sat_current(fitted_params['I_s'], fitted_params['Eg'], T)
            V_i = V_data[i] if isinstance(V_data, list) else V_data
            I_i = I_data[i] if isinstance(I_data, list) else I_data
            I_fit = model.compute_current(V_i, local_params, T=T)
            plt.figure()
            plt.semilogy(V_i, I_i, label=f"Data {T}K")
            plt.semilogy(V_i, I_fit, label=f"Fit {T}K")
            plt.title(f"Diode I-V curve at {T} K")
            plt.legend()
            
            if filename is not None:
                plt.savefig(f"{filename}_{T}K", dpi=300, bbox_inches='tight')
            
        plt.show()
    
def diode_error_plot(V_data, I_data, model, fitted_params, filename=None, temps=None):
    """
    Generates a relative error plot for each voltage point using the I-V data and the fitted parameters

    Args:
        V_data (scalar/numpy array): voltage from I-V data
        I_data (scalar/numpy array): current from I-V data
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        filename (optional): filename for saving the error plot locally, defaults to None
        temps (optional): array of temperatures for multi-temp diode curves, defaults to None
    """
    if temps is None:
        I_fit = model.compute_current(V_data, fitted_params)
        err = (I_fit - I_data) / np.maximum(np.abs(I_data), 1e-15)
        plt.figure()
        plt.plot(V_data, err, label='Relative error')
        plt.legend()
        plt.xlabel("Voltage [V]")
        plt.ylabel("Relative error")
    
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        for i, T in enumerate(temps):
            local_params = fitted_params.copy()
            if 'Eg' in fitted_params:
                local_params['I_s'] = model.compute_sat_current(fitted_params['I_s'], fitted_params['Eg'], T)
            V_i = V_data[i] if isinstance(V_data, list) else V_data
            I_i = I_data[i] if isinstance(I_data, list) else I_data
            I_fit = model.compute_current(V_i, local_params, T=T)
            err = (I_fit - I_i) / np.maximum(np.abs(I_i), 1e-15)
            
            plt.figure()
            plt.plot(V_i, err, label=f'Relative error {T}K')
            plt.legend()
            plt.xlabel("Voltage [V]")
            plt.ylabel("Relative error")
            plt.title(f"Diode fit relative error at {T} K")
            
            if filename is not None:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
        
            plt.show()
            
def diode_sat_current_plot(temps, model, fitted_params, filename=None):
    """
    Plots the saturation current as a function of temperature

    Args:
        temps (list): temperature list
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        filename (optional): filename for saving the error plot locally, defaults to None
    """
    Is = fitted_params['I_s']
    Eg = fitted_params['Eg']
    Is_vals = [model.compute_sat_current(Is, Eg, T) for T in temps]
    
    plt.figure()
    plt.semilogy(temps, Is_vals, label='Fitted $I_s(T)$')
    plt.xlabel("Temperature [K]")
    plt.ylabel("Saturation current [A]")
    plt.legend()
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.show()
    
def plot_mosfet_fit(V_gs, I_data, model, fitted_params, filename=None, yscale='linear'):
    """
    Generates a comparison of the Id-Vgs data and the Id-Vgs curve from the fitted parameters

    Args:
        V_gs (scalar/numpy array): gate-to-source voltage
        I_data (scalar/numpy array): curret from Id-Vgs data
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        filename (str, optional): filename for saving the error plot locally, defaults to None
        yscale (str, optional): 'linear' or 'log', defaults to 'linear'
    """
    I_fit = model.compute_current(V_gs, fitted_params)
    
    if yscale == 'log':
        plt.semilogy(V_gs, I_data, label='Original data')
        plt.semilogy(V_gs, I_fit, label='Fitted data')
        plt.title('Id-Vgs curve on semilog scale')
    else:
        plt.plot(V_gs, I_data, label='Original_data')
        plt.plot(V_gs, I_fit, label='Fitted data')
        plt.title('Id-Vgs curve on linear scale')
    
    plt.xlabel("V_gs [V]")
    plt.ylabel("I_d [A]")
    plt.legend()
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.show()
    
def plot_mosfet_multi(V_ds, I_data, model, fitted_params, V_gs_label=None, filename=None):
    """
    Generates a comparison of the Id-Vds data and the Id-Vds curves from the fitted parameters

    Args:
        V_ds (scalar/numpy array): drain-to-source voltage
        I_data (scalar/numpy array): curret from Id-Vgs data
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        V_gs_label (str, optional): label for title of plots
        filename (str, optional): filename for saving the error plot locally, defaults to None
    """
    I_fit = model.compute_current(fitted_params['vgs_array'], {**fitted_params, 'V_ds': V_ds})
    
    plt.semilogy(V_ds, I_data, label='Original data')
    plt.semilogy(V_ds, I_fit, label='Fitted data')
    
    if V_gs_label is not None:
        plt.title(f"Id-Vd at V_gs = {V_gs_label} V")
    
    plt.xlabel("V_ds [V]")
    plt.ylabel("I_d [A]")
    plt.legend()
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.show()