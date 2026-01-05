import numpy as np
from scipy.constants import e as q_e, k as k_B

from src.models import DiodeModel, MOSFETModel
from src.extraction import ModelExtractor

## Single-temperature fit diode test

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

## Multi-temperature fit diode test

T_ref = 300.0
true_diode_temp = {
    'I_s': 1e-10,
    'Eg': 1.12,
    'n': 1.5,
    'R_s': 2.5
}

def Is_at_T(T):
    return true_diode_temp['I_s'] * (T / T_ref)**3 * np.exp(((true_diode_temp['Eg'] * q_e) / k_B) * (1/T_ref - 1/T))

temps = [280, 300, 320, 340]
datasets = []
V_sweep = np.linspace(0, 0.8, 50)

for T in temps:
    Is_local = Is_at_T(T)
    local_params = {'I_s': Is_local, 'n': true_diode_temp['n'], 'R_s': true_diode_temp['R_s']}
    I_ideal = model.compute_current(V_sweep, local_params, T=T)
    I_noise = I_ideal * (1 + np.random.normal(0, 0.01, size=I_ideal.shape))
    datasets.append((V_sweep, I_noise, T))

diode_temp_guess = {'I_s': 1e-11, 'Eg': 1.0, 'n': 1.2, 'R_s': 0.1}
report_diode_temp = extractor.diode_temp_fit(datasets, initial_params=diode_temp_guess)
fit_diode_temp = report_diode_temp['parameters']

print("True diode temp:", true_diode_temp)
print("Fitted diode temp:", fit_diode_temp)

assert np.abs((fit_diode_temp['I_s'] - true_diode_temp['I_s']) / true_diode_temp['I_s']) < 0.2
assert np.abs((fit_diode_temp['Eg'] - true_diode_temp['Eg']) / true_diode_temp['Eg']) < 0.1
assert np.abs((fit_diode_temp['n'] - true_diode_temp['n']) / true_diode_temp['n']) < 0.1
assert np.abs((fit_diode_temp['R_s'] - true_diode_temp['R_s']) / true_diode_temp['R_s']) < 0.1
print("Diode temp fit passed.\n")