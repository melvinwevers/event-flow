import os
import pickle
from datetime import datetime

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import mpl_toolkits.axes_grid1.inset_locator as mpl_il
from matplotlib.colors import ListedColormap

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, average


# matplotlib settings
scale = 1.2

plt.rcParams.update({"text.usetex": False,
                    "font.family": "Times New Roman",
                    "font.serif": "serif",
                    "mathtext.fontset": "cm",
                    "axes.unicode_minus": False,
                    "axes.labelsize": 9*scale,
                    "xtick.labelsize": 9*scale,
                    "ytick.labelsize": 9*scale,
                    "legend.fontsize": 9*scale,
                    'axes.titlesize': 14,
                    "axes.linewidth": 1
                    })


def get_linkage_mat(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    
    return linkage_matrix



if __name__ == '__main__':

    output_dir = os.path.join('notebooks', 'output', 'clusters', 'grid_search')

    # load trained
    with open(os.path.join(output_dir, 'X.pcl'), 'rb') as fin:
        X = pickle.load(fin)

    with open(os.path.join(output_dir, 'model_full.pcl'), 'rb') as fin:
        mdl_c = pickle.load(fin)

    with open(os.path.join(output_dir, 'model_best.pcl'), 'rb') as fin:
        mdl_d = pickle.load(fin)

    with open(os.path.join(output_dir, 'signal_n_dba.pcl'), 'rb') as fin:
        res_raw = pickle.load(fin)
    

    Z = get_linkage_mat(mdl_c)

        # OPTION 2
    # full dendrogram + clusters at the bottom
    fig = plt.figure(figsize=(15, 6), constrained_layout=True) 
    gs = gridspec.GridSpec(2, 5, figure=fig) 

    alpha_signal = 0.05
    y_axis_limits = [-3, 3]
    x_axis_ticks = list(range(0, 56+14, 14))
    x_axis_ticks_labels = list(range(-28, 28+14, 14)) 


    # cmap = ['#704e2e', '#8cbcb9', '#dfa06e', '#2c5784', '#709176']
    cmap_base = ['#191b27', '#759bab', '#4f5b6f', '#9B7E46', '#AF8D86']
    cmap_light = ['#3c425f', '#90aebb', '#8491A8', '#b89b61', '#bfa5a0']
    cmap_dark = ['#191b27', '#577D8E', '#363E4C', '#9B7E46', '#956C64']
    hierarchy.set_link_color_palette(cmap_base)

    ax0 = plt.subplot(gs[0, :])
    dendrogram(
        Z,
        color_threshold=2.5,
        # truncate_mode='lastp', p=5,
        labels=mdl_d.labels_,
        leaf_font_size=5,
        # no_labels=True,
        above_threshold_color="grey",
        ax=ax0
        )

    ax0.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)


    # CLUSTER 4
    nl_cl = 4
    ax1 = plt.subplot(gs[-1, 0])

    for series in res_raw[nl_cl]['cluster_signals']:
        ax1.plot(series, alpha=alpha_signal, color=cmap_light[0])
    ax1.plot(res_raw[nl_cl]['dba'], color=cmap_dark[0])
    # ax1.title.set_text(f'Cluster {nl_cl+1}')
    ax1.title.set_text('Cluster 1')
    ax1.set_ylim(y_axis_limits)
    ax1.set_xticks(x_axis_ticks)
    ax1.set_xticklabels(x_axis_ticks_labels)

    # ax1.tick_params(left=False,
    #                 bottom=False,
    #                 labelleft=False,
    #                 labelbottom=False)

    # CLUSTER 1
    nl_cl = 1
    ax2 = plt.subplot(gs[-1, -4])

    for series in res_raw[nl_cl]['cluster_signals']:
        ax2.plot(series, alpha=alpha_signal, color=cmap_light[1])
    ax2.plot(res_raw[nl_cl]['dba'], color=cmap_dark[1])
    # ax2.title.set_text(f'Cluster {nl_cl+1}')
    ax2.title.set_text('Cluster 2')
    ax2.set_ylim(y_axis_limits)
    ax2.set_xticks(x_axis_ticks)
    ax2.set_xticklabels(x_axis_ticks_labels)

    # ax2.tick_params(left=False,
    #                 bottom=False,
    #                 labelleft=False,
    #                 labelbottom=False)


    # CLUSTER 2
    nl_cl = 2
    ax3 = plt.subplot(gs[-1, -3])

    for series in res_raw[nl_cl]['cluster_signals']:
        ax3.plot(series, alpha=alpha_signal, color=cmap_light[2])
    ax3.plot(res_raw[nl_cl]['dba'], color=cmap_dark[2])
    # ax3.title.set_text(f'Cluster {nl_cl+1}')
    ax3.title.set_text('Cluster 3')
    ax3.set_ylim(y_axis_limits)
    ax3.set_xticks(x_axis_ticks)
    ax3.set_xticklabels(x_axis_ticks_labels)

    # ax3.tick_params(left=False,
    #                 bottom=False,
    #                 labelleft=False,
    #                 labelbottom=False)


    # CLUSTER 3
    nl_cl = 3
    ax4 = plt.subplot(gs[-1, -2])

    for series in res_raw[nl_cl]['cluster_signals']:
        ax4.plot(series, alpha=alpha_signal, color=cmap_light[3])
    ax4.plot(res_raw[nl_cl]['dba'], color=cmap_dark[3])
    # ax4.title.set_text(f'Cluster {nl_cl+1}')
    ax4.title.set_text('Cluster 4')
    ax4.set_ylim(y_axis_limits)
    ax4.set_xticks(x_axis_ticks)
    ax4.set_xticklabels(x_axis_ticks_labels)

    # ax4.tick_params(left=False,
    #                 bottom=False,
    #                 labelleft=False,
    #                 labelbottom=False)


    # CLUSTER 0
    nl_cl = 0
    ax5 = plt.subplot(gs[-1, -1])

    for series in res_raw[nl_cl]['cluster_signals']:
        ax5.plot(series, alpha=alpha_signal, color=cmap_light[4])
    ax5.plot(res_raw[nl_cl]['dba'], color=cmap_dark[4])
    # ax5.title.set_text(f'Cluster {nl_cl+1}')
    ax5.title.set_text('Cluster 5')
    ax5.set_ylim(y_axis_limits)
    ax5.set_xticks(x_axis_ticks)
    ax5.set_xticklabels(x_axis_ticks_labels)

    # ax5.tick_params(left=False,
    #                 bottom=False,
    #                 labelleft=False,
    #                 labelbottom=False)

    plt.tight_layout()

    outdir = os.path.join('notebooks', 'output', 'clusters')
    daytag = datetime.today().strftime('%Y%m%d')
    fname = f'{daytag}_cluster_dendrogram.png'

    plt.savefig(
        os.path.join(outdir, fname)
        )