"""
Plot data
"""
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

from matplotlib import font_manager


if __name__ == '__main__':
    omsignal_df = pd.read_csv('om_signal.csv', index_col=0).dropna()
    fitbit_df = pd.read_csv('fitbit.csv', index_col=0).dropna()
    owl_in_one_df = pd.read_csv('owl_in_one.csv', index_col=0).dropna()
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    ticks_font = font_manager.FontProperties(size=11)
    
    sns.distplot(np.array(fitbit_df), kde=False, rug=False, norm_hist=True, ax=axs[0], color="y")
    sns.distplot(np.array(omsignal_df), kde=False, rug=False, norm_hist=True, ax=axs[1], color="b")
    sns.distplot(np.array(owl_in_one_df), kde=False, rug=False, norm_hist=True, ax=axs[2], color="r")
    
    for i in range(3):
        for label in axs[i].get_xticklabels():
            label.set_fontproperties(ticks_font)
        
        for label in axs[1].get_yticklabels():
            label.set_fontproperties(ticks_font)

        axs[i].set_ylabel('Relative Frequency', fontsize=13)
        axs[i].set_xlim(0, 25)
        axs[i].set_xlabel('Sensor Usage Time Per Day (hour)', fontsize=13)

    axs[0].set_ylim(0, 0.16)
    axs[1].set_ylim(0, 0.3)
    axs[2].set_ylim(0, 0.3)

    axs[0].set_title('Fitbit Usage Histogram', fontsize=13, fontweight="bold")
    axs[1].set_title('OM Signal Usage Histogram', fontsize=13, fontweight="bold")
    axs[2].set_title('Jelly Pro Usage Histogram', fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    plt.show()
