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