
import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import pickle as pickle
from struct import unpack
import scipy.stats as stats
from brian2 import *
import pandas as pd
import glob

FOLDER = "/home/smestern/Dropbox/brian2_SDTP/stdp-mnist-brian2/E_TO_E_no_XE_400/"

#glob the _invitro
invitro = glob.glob(FOLDER + "/**/_invitro_cross_val_scores.csv", recursive=True)

#load the invitro
dict_results = {}
for file in invitro:
    #print(file)
    invitro = np.genfromtxt(file, delimiter=',')
    folder_name = os.path.dirname(file)
    dict_results[folder_name] = {}
    [dict_results[folder_name].update({f'trial {i}': item}) for i, item in enumerate(invitro.tolist())]
    #FROM THE SAME FORLDER LOAD
    insilco = np.genfromtxt(folder_name + "/neuron_cross_val.csv", delimiter=',')[2:, 1:]

    [dict_results[folder_name].update({f'insilico trial {i} mean':np.nanmean(item)}) for i, item in enumerate(insilco.T)]
    [dict_results[folder_name].update({f'insilico trial {i} std':np.nanstd(item)}) for i, item in enumerate(insilco.T)]

    #run a one way anova on each row
    vals = [stats.f_oneway(insilco[i, :], invitro) for i in range(insilco.shape[0])]
    #combine the p_values
    p_values = np.array([val[1] for val in vals])
    p_values = np.nan_to_num(p_values)
    p_values = stats.combine_pvalues(p_values, method='fisher')[1]

    dict_results[folder_name].update({'pvalues':p_values})

#turn the dict into a dataframe, and save it
df = pd.DataFrame.from_records(dict_results)
df.to_csv(FOLDER + "invitro_cross_val_scores.csv", sep=',', encoding='utf-8')