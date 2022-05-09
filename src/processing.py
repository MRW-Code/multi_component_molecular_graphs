import pandas as pd

def load_lit_dataset(path): return pd.read_csv(path)

def get_useful_columns(dataset_path):
    df = load_lit_dataset(dataset_path)
    to_keep = ['solute_smiles',
               'solvent_smiles',
               'solubility_g_100g']
    df = df.loc[:, to_keep]
    return df