import os
import sys
import pandas as pd
import numpy

if __name__ == "__main__":
    fitbit_folder = os.path.join('../../data/keck_wave2/3_preprocessed_data/fitbit')
    print(os.listdir(fitbit_folder))
    
    for file in os.listdir(fitbit_folder):
        if 'heartRate.csv' in file:
            data_df = pd.read_csv(os.path.join(fitbit_folder, file))
            data_df = data_df[:-1]
            # data_df = data_df.sort_index()
            data_df.to_csv(os.path.join('../../data/keck_wave2/3_preprocessed_data/', 'fit_correction', file), index=False)
            print(data_df)
            