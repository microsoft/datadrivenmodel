import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#sns.set()

fname_ddm = './outputs/2021-07-09/23-28-15/logs/2021-07-09-23-28-17_DDM_log.csv'
fname_sim = './outputs/2021-07-09/23-28-15/logs/2021-07-09-23-28-17_SIMDATA_log.csv'
col = ['state_theta', 'state_alpha', 'action_Vm']

def plot_diff(fname_ddm, fname_sim, col):
    df_ddm = pd.read_csv(fname_ddm)
    df_sim = pd.read_csv(fname_sim)

    fig = plt.figure(figsize=(10,8))
    numSubPlots = len(col)
    
    ax1 = plt.subplot(numSubPlots, 1, 1)
    plt.plot(df_ddm['iteration'], df_ddm[col[0]], label=col[0]+'_ddm')
    plt.plot(df_sim['iteration'], df_sim[col[0]], label=col[0]+'_sim')
    plt.xticks(rotation='horizontal')
    plt.legend(loc='upper right')

    for i in range(1,numSubPlots):
        if 'action' in col[i]:
            ax2 = plt.subplot(numSubPlots, 1, i+1, sharex=ax1)
            plt.plot(df_ddm['iteration'][1:], df_ddm[col[i]][1:], label=col[i]+'_ddm')
            plt.plot(df_sim['iteration'][1:], df_sim[col[i]][1:], label=col[i]+'_sim')
            plt.xticks(rotation='horizontal')
            plt.legend(loc='upper right')
        else:
            ax2 = plt.subplot(numSubPlots, 1, i+1, sharex=ax1)
            plt.plot(df_ddm['iteration'], df_ddm[col[i]], label=col[i]+'_ddm')
            plt.plot(df_sim['iteration'], df_sim[col[i]], label=col[i]+'_sim')
            plt.xticks(rotation='horizontal')
            plt.legend(loc='upper right')

    plt.show()

if __name__ == '__main__':
    plot_diff(fname_ddm, fname_sim, col)