# Shockley equation: I = I_s * (exp(V / (n * V_t)) - 1)
# I_s: saturation current
# V: applied voltage
# n: ideality factor
# V_t: threshold voltage

import numpy as np
from scipy.constants import k as k_B, e as q_e

class DiodeModel:
    def __init__(self, T=300):
        """
        Class constructor for a generic diode device at room temperature

        Args:
            T (int, optional): temperature, defaults to 300.
        """
        self.temp = T
        
    def compute_current(self, V, params, T=None):
        """
        Comptue diode current using Shockley equation with Newton-Raphson iteration to account for series resistance

        Args:
            V (scalar/numpy array): applied voltage
            params (dict): model parameters including saturation current, ideality, and series resistance
            T (float, optional): temperature in Kelvin, defaults to model's temperature if None

        Returns:
            Numpy array: calculated current at each voltage point
        """
        I_s = params['I_s']
        n = params['n']
        R_s = params.get('R_s', 0.0)
        
        if T is None:
            T = self.temp
            
        V = np.asarray(V)
        
        def solve_current(V):
            I = 0.0
            
            for _ in range(50):
                Vd = V - I * R_s
                arg = np.clip(q_e * Vd / (n * k_B * T), -50, 50) # prevent exponential overflow
                f_val = I_s * (np.exp(arg) - 1) - I
                df_val = -(I_s * np.exp(arg) * R_s * q_e / (n * k_B * T)) - 1
                
                if abs(df_val) < 1e-15:
                    break
                
                I_new = I - f_val / df_val
                
                if abs(I_new - I) < 1e-12:
                    return I_new
                
                I = I_new
                
            return I
        
        solve_vec = np.vectorize(solve_current, otypes=[float])(V)
        return solve_vec
    
    def get_param_bounds(self):
        """
        Returns standard bounds for device parameter

        Returns:
            Dict: standard saturation current (1e-16 to 1e-6 A) and ideality factor range (1.0 to 2.0)
        """
        return {
            'I_s': (1e-16, 1e-6),
            'Eg': (0.1, 5.0),
            'n': (1.0, 2.0),
            'R_s': (0.0, 10.0)
        }