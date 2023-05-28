#deepwalk result
import numpy as np

from GraphEmbedding_master.ge.classify import read_node_label, Classifier
from GraphEmbedding_master.ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score

def evaluate_embeddings(embeddings):
    #X, Y = read_node_label('../data/drug/label.txt')
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)
    # x_test,ytest =
    # clf.predict()


def plot_embeddings(embeddings,):
    #X, Y = read_node_label('../data/drug/label.txt')
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    #
    # model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    # model.train(window_size=5, iter=3)
    # embeddings = model.get_embeddings()
    # print(len(embeddings))#2405
    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
    G = nx.read_edgelist('../data/drug/edge.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = DeepWalk(G, walk_length=5, num_walks=881, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    X_train = []
    Y_train = []
    with open("../data/drug/edge.txt", 'r', encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            drug1, drug2 = line.split(" ")[0], line.split(" ")[1][:-1]
            X_train.append((embeddings[drug1] + embeddings[drug2]) / 2)
        f.close()
    num, Y_train = read_node_label('../data/drug/label.txt')

    X_test = []
    Y_test = []
    with open("../data/drug/test_edge.txt", 'r', encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            drug1, drug2 = line.split(" ")[0], line.split(" ")[1][:-1]
            X_test.append((embeddings[drug1] + embeddings[drug2]) / 2)
        f.close()
    num2, Y_test = read_node_label('../data/drug/test_label.txt')

    k = 5
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,Y_train)

    Y_pred = clf.predict(X_test)

    print(classification_report(Y_test, Y_pred))
    print("-------------------------")
    mcc = matthews_corrcoef(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print("MCC:", mcc)
    print("AUC:", auc)

    # print(len(embeddings))#1017
    # print("----------------")
    # #print(embeddings)
    # print("----------------")
    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
    # arr = []
    # #with open('../data/wiki/Wiki_edgelist.txt', "r", encoding='UTF-8') as f :
    # with open('../data/drug/edge.txt', "r", encoding='UTF-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         drug1, drug2 = line.split(" ")[0],line.split(" ")[1][:-1]
    #         print(drug1,drug2)
    #         if drug1 not in arr:
    #             arr.append(drug1)
    #         if drug2 not in arr:
    #             arr.append(drug2)
    # print(len(arr))