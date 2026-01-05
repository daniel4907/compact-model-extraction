import numpy as np
from scipy.constants import e as q_e, k as k_B

from src.models import DiodeModel, MOSFETModel
from src.extraction import ModelExtractor

## MOSFET fit test

model = MOSFETModel()
extractor = ModelExtractor(model)

true_params = {'V_th': 0.67, 'k_n': 5e-4, 'lam': 0.02, 'V_ds': 1.0}
V_gs = np.linspace(0, 2, 100)
I_true = model.compute_current(V_gs, true_params)
I_noise = I_true * (1 + np.random.normal(0, 0.02, size=I_true.shape))

report = extractor.mosfet_fit(V_gs, I_noise, V_ds=true_params['V_ds'])
fit = report['parameters']

print("True MOSFET params:", true_params)
print("Fit MOSFET params:", fit)

assert np.abs((fit['V_th'] - true_params['V_th']) / true_params['V_th']) < 0.1
assert np.abs((fit['k_n'] - true_params['k_n']) / true_params['k_n']) < 0.1
# assert np.abs((fit['lam'] - true_params['lam']) / true_params['lam']) < 0.1 Lambda checking on single fit is inaccurate due to lack of data
assert np.abs((fit['V_ds'] - true_params['V_ds']) / true_params['V_ds']) < 0.1
print("MOSFET fit passed.\n")

## MOSFET global fit test

global_params = {
    'V_th': 0.8,
    'k_n': 1e-3,
    'lam': 0.05
}

vgs_sweep = [1.5, 2.0, 2.5, 3.0]
vds_sweep = np.linspace(0, 5.0, 50)
datasets = []

for vgs in vgs_sweep:
    curve_params = global_params.copy()
    curve_params['V_ds'] = vds_sweep
    vgs_array = np.full_like(vds_sweep, vgs)
    I_ideal = model.compute_current(vgs_array, curve_params)
    I_noise = I_ideal * (1 + np.random.normal(0, 0.02, size=I_ideal.shape))
    datasets.append((vds_sweep, I_noise, vgs))
    
initial_guess = {'V_th': 0.5, 'k_n': 1e-4, 'lam': 0.0}
report = extractor.multi_mosfet_fit(datasets, initial_params=initial_guess)
fit = report['parameters']

print("True global MOSFET params:", global_params)
print("Fitted global MOSFET params:", fit)

assert np.abs((fit['V_th'] - global_params['V_th']) / global_params['V_th']) < 0.1
assert np.abs((fit['k_n'] - global_params['k_n']) / global_params['k_n']) < 0.1
assert np.abs((fit['lam'] - global_params['lam']) / global_params['lam']) < 0.1
print("MOSFET global fit passed.\n")