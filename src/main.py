from preprocessing import construct_optimizer_list, make_test_edges
from input_data import load_data
import numpy as np
import sys,getopt

from trainGcn import train_gcn
from trainNN import generate_data,train_nn

def train(dataset, weightRate):
    adj, features, falseEdges = load_data(dataset)

    #generate training and test data
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = make_test_edges(weightRate, adj, falseEdges)

    print adj_train.shape
    print train_edges.shape,train_edges_false.shape

    #embeddings returned by W-VGAE
    emb = train_gcn(features,adj_train,train_edges,train_edges_false,test_edges,test_edges_false)

    #generate paired training and test data, similar to GCN
    X_train,Y_train = generate_data(emb, train_edges, train_edges_false)
    X_test,Y_test = generate_data(emb, test_edges, test_edges_false)

    #the final softmax classifier
    acc = train_nn(X_train,Y_train,X_test,Y_test)
    print 'accuracy:',acc[0]
    print 'sensitivity:',acc[1]
    print 'specificity:',acc[2]
    print 'precision:',acc[3]

def main():
    opts, args = getopt.getopt(sys.argv[1:],"d:w:",["dataset=","wr="])
    dataset = "Hprd"
    weightRate = 1
    for opt, arg in opts:
        if opt == '--dataset':
            dataset = arg
        if opt == '--wr':
            weightRate=int(arg)

    print dataset
    print weightRate
    train(dataset,weightRate)

if __name__ == "__main__":
    main()
