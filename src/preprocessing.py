import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

def construct_optimizer_list(num_node, posEdges, falseEdges):
    mask_index = []
    for x in posEdges:
        temp = x[0] * num_node + x[1]
        mask_index.append(temp)
    for x in falseEdges:
        temp = x[0] * num_node + x[1]
        mask_index.append(temp)

    return np.array(mask_index)

def make_test_edges(weightRate, adj, falseEdges):
    # Function to build test set with 10% positive links and 10% highly negative links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    posNum = edges.shape[0]
    num_test = int(np.floor(posNum / 10.))
    num_train = edges.shape[0] - num_test

    #generate positive sets
    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    train_edge_idx = all_edge_idx[:num_train]
    train_edges = edges[train_edge_idx]
    test_edge_idx = all_edge_idx[num_train:(num_train + num_test)]
    test_edges = edges[test_edge_idx]
    
    falseNum = falseEdges.shape[0]
    num_neg_test = int(np.floor(falseNum/10.))
    num_neg_train = falseNum - num_neg_test

    #generate negative sets
    all_edge_neg_idx = range(falseNum)
    np.random.shuffle(all_edge_neg_idx)
    train_edge_neg_idx = all_edge_neg_idx[:num_neg_train]
    test_edge_neg_idx = all_edge_neg_idx[num_neg_train:(num_neg_train+num_neg_test)]
    train_edges_false = falseEdges[train_edge_neg_idx]
    test_edges_false = falseEdges[test_edge_neg_idx]

    # Re-build training adjacency matrix
    adj_train = np.zeros(adj.shape)
    facNeg = -0.001
    facPos = weightRate * facNeg

    for x in train_edges:
        adj_train[x[0]][x[1]] = facPos
        adj_train[x[1]][x[0]] = facPos
    for x in train_edges_false:
        adj_train[x[0]][x[1]] = facNeg
        adj_train[x[1]][x[0]] = facNeg
    adj_train = sp.csr_matrix(adj_train)


    return adj_train, train_edges, train_edges_false, test_edges, test_edges_false


