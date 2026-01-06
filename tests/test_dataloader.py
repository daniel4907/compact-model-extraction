import numpy as np
import os, pytest

from src.dataloader import DataLoader
from src.models import MOSFETModel
from src.extraction import ModelExtractor
from src.utils import *

# generate gold test data
# --- Test Single MOSFET Curve Generation and Loading ---

single_curve_params = {'V_th': 1.0, 'k_n': 1e-2, 'lam': 0.05}

# 1. Transfer Curve (Id-Vgs)
transfer_path = 'data/test_mosfet_transfer.csv'
vgs_sweep = np.linspace(0, 3, 50) # Sweep 0 to 3V, crossing Vth=1.0
generate_mosfet_csv(transfer_path, single_curve_params, sweep_type='Id-Vgs', sweep=vgs_sweep, val=1.0)

# 2. Output Curve (Id-Vds)
output_path = 'data/test_mosfet_output.csv'
vds_sweep = np.linspace(0, 5, 50)
generate_mosfet_csv(output_path, single_curve_params, sweep_type='Id-Vds', sweep=vds_sweep, val=3.0) # Vgs=3.0 > Vth=1.0

# multi-curve MOSFET data

filepath = 'data/test_mosfet_data.csv'
true_params = {'V_th': 3.0, 'k_n': 1e-2, 'lam': 0.3}
vgs_sweep = [1.0, 2.0, 3.0, 4.0, 5.0]
vds_sweep = np.linspace(0, 10, 50)

generate_multi_mosfet_csv(filepath, true_params, vgs_sweep, vds_sweep)

# single-temperature diode data

filepath = 'data/test_diode_data.csv'
true_params = {'I_s': 1e-10, 'n': 1.5, 'R_s': 2.5}
v_sweep = np.linspace(0, 5, 50)
T = 300

generate_diode_csv(filepath, true_params, v_sweep, T=T)

# multi-temperature diode data

filepath = 'data/test_multitemp_diode_data.csv'
true_params = {'I_s': 1e-10, 'n': 1.5, 'R_s': 2.5}
v_sweep = np.linspace(0, 5, 50)
temps = [280, 300, 320, 340]

generate_multitemp_diode_csv(filepath, true_params, v_sweep, temps)

# test loading CSV and mapping columns

loader = DataLoader()
filepath = "data/test_mosfet_data.csv"
col_map = {
    'V_Gate': 'V_gs',
    'V_Drain': 'V_ds',
    'I_Drain': 'I_d'
}

df = loader.load_csv(filepath, col_map=col_map)

assert 'V_gs' in df.columns
assert 'V_ds' in df.columns
assert 'I_d' in df.columns

# test converting dataframe into format used by ModelExtractor

datasets = loader.get_mosfet_datasets()
vds, ids, vgs = datasets[0]

assert isinstance(vds, np.ndarray)
assert isinstance(ids, np.ndarray)
assert isinstance(vgs, float)

# test integration with ModelExtractor

model = MOSFETModel()
extractor = ModelExtractor(model)

initial_guess = {
    'V_th': 0.5,
    'k_n': 1e-4,
    'lam': 0.0
}

report = extractor.multi_mosfet_fit(datasets, initial_params=initial_guess)

assert report['success'] is True
print ("Fitted params from CSV:", report['parameters'])