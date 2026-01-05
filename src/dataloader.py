import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self):
        """
        DataLoader constructor
        """
        self.df = None
        self.req_cols = []
        
    def load_csv(self, filepath, col_map=None, comment_char='!'):
        """
        Load I-V data from a CSV file

        Args:
            filepath (str): path to file
            col_map (dict, optional): mapping from file columns to standard names, e.g. {V_Gate: 'V_gs'}, defaults to None
            comment_char (str, optional): character that indicates comments to skip

        Raises:
            FileNotFoundError: in case filepath is invalid
            ValueError: error when reading the CSV or when column not found in the CSV
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try: # read through the CSV, skip comments
            self.df = pd.read_csv(filepath, comment=comment_char)
        except Exception as e:
            raise ValueError(f"Error reading CSV: {e}")
        
        self.df.columns = self.df.columns.str.strip() # strip whitespace from headers
        
        if col_map: # apply column mapping if included in args
            for file_col, std_col in col_map.items():
                if file_col not in self.df.columns:
                    raise ValueError(f"Column '{file_col}' not found in file. Available: {list(self.df.columns)}")
                
            self.df.rename(columns=col_map, inplace=True)
            
        return self.df
    
    def filter_compliance(self, current_col='I_d', limit=0.1):
        """
        Remove points where current hit a compliance limit
        """
        if self.df is not None and current_col in self.df.columns:
            self.df = self.df[self.df[current_col] < limit]
            
    def get_mosfet_datasets(self, vgs_col='V_gs', vds_col='V_ds', id_col='I_d'):
        """
        Splits dataframe into list of datasets to be parsed by ModelExtractor

        Returns:
            datasets (list): list of tuples (Vds_array, Id_array, Vgs)
        """
        if self.df is None:
            raise ValueError("No data loaded, call load_csv() first")
        
        datasets = []
        grouped = self.df.groupby(vgs_col)
        
        for vgs, group in grouped:
            group = group.sort_values(vds_col)
            vds_data = group[vds_col].values
            id_data = group[id_col].values
            datasets.append((vds_data, id_data, float(vgs)))
            
        return datasets