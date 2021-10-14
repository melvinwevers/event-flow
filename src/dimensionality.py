import networkx
import pandas as pd

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def transform_to_ajencency_mat(df):
    '''Transform distance matrix to adjecency matrix
    '''
    df['distance'] = df["distance"].apply(lambda x: float(x))

    # Turn into edgelist
    edgeList = df.values.tolist()

    # Create empty graph
    G = networkx.Graph()

    # Add edges to graph
    for i in range(len(edgeList)):
        G.add_edge(edgeList[i][1], edgeList[i][2], weight=edgeList[i][0])

    # Get order of nodes
    nodes = list(G.nodes)
    # nodes = [int(x) for x in nodes]
    nodes = [x for x in nodes]
    nodes = sorted(list(set(nodes)))

    # Get adjacency matrix
    A = networkx.adjacency_matrix(G, nodelist=nodes).A

    return A, nodes


def fit_projection(df, projection='umap', **kwargs):
    '''Project adjecency matrix to 2D.
    '''

    adjecency_matrix, nodes = transform_to_ajencency_mat(df)

    if projection == 'umap':
        proj = umap.UMAP(**kwargs, random_state=42).fit_transform(adjecency_matrix)
    elif projection == 'tsne':
        proj = TSNE(n_components=2, random_state=42).fit_transform(adjecency_matrix)
    elif projection == 'pca':
        proj = PCA(n_components=2).fit_transform(adjecency_matrix)

    return pd.DataFrame({'X': proj[:, 0], 'Y': proj[:, 1], 'node': nodes})
