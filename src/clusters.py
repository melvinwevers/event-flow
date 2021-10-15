'''
Cluster data and compare group differences
a) k-means parameter grid search
b) hierarchical clustering exploratory
'''
import os
import time
import argparse

import pandas as pd
import numpy as np
import ndjson

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


def grid_search_clustering(X, parameters, algorithm, use_silhouette=True, res_path=None):
    '''Given parameter space, fit many kmeans models 
    and track performance.

    Parameters
    ----------
    X : np.array
        Training data.
    parameters : dict
        Named parameters, where key is parameter name
        and value a list of values to try.
    algorithm : instance
        tag of algorithm to use for the clustering.
        Options are kmeans & agglomerative are implemented. 
    use_silhouette : bool
        Calculate silhouette score for each model?
        If false, only inertia will be included in output.
    res_path : str, optional
        If specified, results will be dumped there, by default None

    Returns 
    -------
    list
        if res_path not specified, returns log of results
    '''

    param_grid = ParameterGrid(parameters)

    # pick cluster algorithm
    grid_search_results = []
    for params in param_grid:
        t0 = time.time()
        km = algorithm(**params)
        km.fit(X)
        t_bench = time.time() - t0

        res = {}

        if use_silhouette:
            silhouette_ = silhouette_score(
                X,
                km.labels_,
                metric='cosine'
            )
            res['silhouette'] = silhouette_

        res['train_time'] = t_bench
        res['params'] = params

        grid_search_results.append(res)

    if res_path:
        with open(res_path, 'w') as fout:
            ndjson.dump(grid_search_results, fout)

    return grid_search_results


def plot_elbow(grid_search_results, metric='silhouette',
               prim_param='n_clusters', sec_param=None, get_df=False):
    '''Make elbow plot comparing kmeans performance under up to 
    two model parameters.

    Parameters
    ----------
    grid_search_results : list of dict
        Results of grid searching.
    metric : str, optional
        Name of _continuous_ metric from results, by default 'silhouette'
    prim_param : str, optional
        Primary model parameter to plot, by default 'n_clusters'
        Shown on X axis.
    sec_param : str, optional
        Secondary model parameter to plot, by default None

    Returns
    -------
    plt.figure
    '''

    # unpack to df
    res = pd.DataFrame(grid_search_results)
    params = res['params'].apply(pd.Series)
    res = res.join(params)

    sns.set_theme(style="whitegrid", color_codes=True)
    fig = plt.figure(figsize=(10, 10))
    ax = sns.lineplot(
        data=res,
        x=prim_param,
        y=metric,
        hue=sec_param,
        palette='muted'
    )

    ax.set_xticks(res[prim_param])

    if get_df:
        return fig, res
    
    else:
        return fig


def plot_learned_clusters(X, k_means):
    '''Make plot showing raw data colored by label,
    and the centroid of each cluster.

    Parameters
    ----------
    X : np.array
        Training data

    k_means : sklearn.cluster._kmeans.KMeans
        Trained instance of

    Returns
    -------
    plt.figure
    '''

    sns.set_theme(style="white", color_codes=True)
    color_palette = sns.color_palette(
        palette='muted', n_colors=k_means.n_clusters)

    fig = sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=k_means.labels_,
        palette=color_palette
    )

    fig.scatter(
        x=k_means.cluster_centers_[:, 0],
        y=k_means.cluster_centers_[:, 1],
        s=200,
        c=color_palette,
        linewidths=2,
        edgecolors='grey'
    )

    fig.set(
        title=f'''
        k-means \
        n_clusters: {k_means.n_clusters} \
        intertia: {round(k_means.inertia_, 2)}
        '''
    )

    return fig



###
### hierarchical clustering
###

def plot_dendrogram(model, **kwargs):
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

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_hierarchy_2D(df, title='Hierarchical Clustering Dendrogram', node_col='newspaper_event'):
    df.index = df[node_col]
    X = np.array(df[['X', 'Y']])

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(X)
    fig = plt.figure(figsize=(20, 10))
    plt.title(title)
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=7, labels=df.index)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    
    return fig
