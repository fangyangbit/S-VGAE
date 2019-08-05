import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import scale,normalize


def load_data(dataset):
    with open('../data/'+dataset+'/allx_ct.pkl') as f:
        allx = pkl.load(f)
    with open('../data/'+dataset+'/PositiveEdges.txt') as f:
        posEdges = f.readlines()
    with open('../data/'+dataset+'/NegativeEdges.txt') as f:
        negEdges = f.readlines()

    #generate highly negative sets
    rows = []
    cols = []
    for line in negEdges:
        x,y = line.split()
        rows.append(int(x))
        cols.append(int(y))

    X = np.array(rows)
    Y = np.array(cols)
    falseEdges = np.vstack((X,Y)).transpose()

    #CT codings for proteins
    features = normalize(allx)

    #generate adjacency matrix
    num_node = allx.shape[0]
    adj = np.zeros((num_node,num_node))
    for line in posEdges:
        x,y = line.split()
        adj[int(x)][int(y)] = 1
        adj[int(y)][int(x)] = 1

    adj = sp.csr_matrix(adj)

    return adj, features, falseEdges
