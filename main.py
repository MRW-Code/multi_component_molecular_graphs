from src.utils import args
import pandas as pd
from src.processing import get_useful_columns

if __name__ == '__main__':
    # Load the literature dataset and drop all the descriptor columns
    lit_d_set = get_useful_columns('./data/dataset_vas_et_al.csv')

    # Convert the SMILES codes into molcular graphs


    # Define and create the model


    # Pytorch training loop


    # Return some metics and or figures


    print('done')