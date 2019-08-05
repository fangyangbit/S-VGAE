from __future__ import division
from __future__ import print_function

import time
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import cPickle as pkl

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, construct_optimizer_list


def train_gcn(features,adj_train, train_edges, train_edges_false, test_edges, test_edges_false):
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 96, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 48, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
    flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
    flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

    model_str = FLAGS.model

    #1-dim index array, used in cost function to only focus on those interactions with high confidence
    mask_index = construct_optimizer_list(features.shape[0], train_edges, train_edges_false)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj_train
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj = adj_train

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float64),
        'adj': tf.sparse_placeholder(tf.float64),
        'adj_orig': tf.sparse_placeholder(tf.float64),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)


    pos_weight = 1
    norm = 1
    #pos_weight = train_edges_false.shape[0] / float(train_edges.shape[0])
    #norm = (train_edges.shape[0]+train_edges_false.shape[0]) / float(train_edges_false.shape[0]*train_edges_false.shape[0])

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          mask=mask_index)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           mask=mask_index)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        print("Epoch:", '%04d' % (epoch+1), "train_loss=", "{:.5f}".format(outs[1]))


    print("Optimization Finished!")
    
    #return embedding for each protein
    emb = sess.run(model.z_mean,feed_dict=feed_dict)
    return emb

