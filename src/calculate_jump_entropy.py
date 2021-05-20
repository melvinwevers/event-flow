import argparse
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from entropies import jsd
import os
import numpy as np
import pandas as pd



def load_model(input_file):
    with open((input_file), "rb") as fobj:
        mdl = pickle.load(fobj)
        theta = mdl["theta"]
    dates = mdl["dates"]
    return theta, dates

def calc_jump(theta, window, j, jumps, meas=jsd, weight=0):
    N_hat = np.zeros(theta.shape[0])
    N_sd = np.zeros(theta.shape[0])
    for i, x in enumerate(theta):
        submat = theta[(i - window):(i+window),]
        submat_jump = theta[((i - window) + j):((i+window)+j),]
        tmp = np.zeros(submat.shape[0])
        if submat.any():
            for ii, (xx, xx_j) in enumerate(zip(submat, submat_jump)):
                tmp[ii] = meas(xx, xx_j)
        else:
            tmp = np.zeros([window]) + weight

        N_hat[i] = np.mean(tmp)
        N_sd[i] = np.std(tmp)
    jumps[j] = N_hat

    return jumps

def make_jump_entropies(input_file, window, jump_range=range(-1500, 1500, 1)):

    theta, dates = load_model(input_file)

    jumps = dict()
    jumps2 = Parallel(n_jobs=7)(delayed(calc_jump)(theta, window, jump, jumps) for jump in tqdm(jump_range))
    jumps = dict()
    # take dictionary out of list output from parallel
    for _ in jumps2:
        for k, v in _.items():
            jumps[k] = v

    filename = os.path.basename(input_file) 
    df_kld = pd.DataFrame.from_dict(jumps, orient='columns')
    df_kld['dates'] = dates
    df_kld.to_pickle(f"../models/jumps/{filename[:-4]}_jump.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--window', type=int, default=7)

    args = parser.parse_args()

    if not os.path.exists('../models/jumps'):
        os.makedirs('../models/jumps')
    
    make_jump_entropies(args.input_file, args.window)

