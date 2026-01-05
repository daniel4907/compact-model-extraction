import numpy as np
import os, pytest

from src.dataloader import DataLoader
from src.models import MOSFETModel
from src.extraction import ModelExtractor
from src.utils import generate_mosfet_csv

# generate gold test data

filepath = 'data/test_mosfet_data.csv'
true_params = {'V_th': 0.7, 'k_n': 0.001, 'lam': 0.02}
vgs_list = [1.0, 2.0, 3.0]
vds_sweep = np.linspace(0, 5, 20)

generate_mosfet_csv(filepath, true_params, vgs_list, vds_sweep)

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