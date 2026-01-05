import numpy as np
from scipy.constants import e as q_e, k as k_B

from src.models import DiodeModel
from src.extraction import ModelExtractor

model = DiodeModel()
extractor = ModelExtractor(model)

params = {
    'I_s': 1e-10, 
    'n': 1.5, 
    'R_s': 2.5
}

V_data = np.linspace(0, 0.8, 50)
T_test = 300
I_data_true = model.compute_current(V_data, params, T=T_test)

np.random.seed(67)
I_data = I_data_true * (1 + np.random.normal(0, 0.02, size=I_data_true.shape))
initial_guess = {'I_s': 1e-11, 'n': 1.4, 'R_s': 0.1}
ls = extractor.diode_fit(V_data, I_data, T=T_test, initial_params=initial_guess)

print("True params:", params)
print("Fit params:", ls['parameters'])

assert(np.abs((ls['parameters']['I_s'] - params['I_s']) / params['I_s']) <= 0.2)
assert(np.abs((ls['parameters']['n'] - params['n']) / params['n']) <= 0.1)
assert(np.abs((ls['parameters']['R_s'] - params['R_s']) / params['R_s']) <= 0.1)
assert((ls['success'] is True))

print("Single fit passed.\n")

T_ref = 300.0
true_global = {
    'I_s': 1e-10,
    'Eg': 1.12,
    'n': 1.5,
    'R_s': 2.5
}

def Is_at_T(T):
    return true_global['I_s'] * (T / T_ref)**3 * np.exp(((true_global['Eg'] * q_e) / k_B) * (1/T_ref - 1/T))

temps = [280, 300, 320, 340]
datasets = []
V_sweep = np.linspace(0, 0.8, 50)

for T in temps:
    Is_local = Is_at_T(T)
    local_params = {'I_s': Is_local, 'n': true_global['n'], 'R_s': true_global['R_s']}
    I_ideal = model.compute_current(V_sweep, local_params, T=T)
    I_noise = I_ideal * (1 + np.random.normal(0, 0.01, size=I_ideal.shape))
    datasets.append((V_sweep, I_noise, T))

global_guess = {'I_s': 1e-11, 'Eg': 1.0, 'n': 1.2, 'R_s': 0.1}
report_global = extractor.global_fit(datasets, initial_params=global_guess)
fit_global = report_global['parameters']

print("True global:", true_global)
print("Fitted global:", fit_global)

assert np.abs((fit_global['I_s'] - true_global['I_s']) / true_global['I_s']) < 0.2
assert np.abs((fit_global['Eg'] - true_global['Eg']) / true_global['Eg']) < 0.1
assert np.abs((fit_global['n'] - true_global['n']) / true_global['n']) < 0.1
assert np.abs((fit_global['R_s'] - true_global['R_s']) / true_global['R_s']) < 0.1
print("Global fit passed.\n")