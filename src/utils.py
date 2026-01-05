import numpy as np
import pandas as pd

from src.models import MOSFETModel

def generate_mosfet_csv(filepath, params, vgs_list, vds_sweep):
    model = MOSFETModel()
    data = []
    for vgs in vgs_list:
        vgs_array = np.full_like(vds_sweep, vgs)
        i_d = model.compute_current(vgs_array, {**params, 'V_ds': vds_sweep})
        for vds, id in zip(vds_sweep, i_d):
            data.append({'V_Gate': vgs, 'V_Drain': vds, 'I_Drain': id})
            
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)