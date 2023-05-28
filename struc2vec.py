import numpy as np



from GraphEmbedding_master.ge.classify import read_node_label,Classifier

from GraphEmbedding_master.ge import Struc2Vec

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score

import matplotlib.pyplot as plt

import networkx as nx

from sklearn.manifold import TSNE



# def evaluate_embeddings(embeddings):
#
#     X, Y = read_node_label('../data/flight/labels-brazil-airports.txt',skip_head=True)
#
#     tr_frac = 0.8
#
#     print("Training classifier using {:.2f}% nodes...".format(
#
#         tr_frac * 100))
#
#     clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
#
#     clf.split_train_evaluate(X, Y, tr_frac)





# def plot_embeddings(embeddings,):
#
#     X, Y = read_node_label('../data/flight/labels-brazil-airports.txt',skip_head=True)
#
#
#
#     emb_list = []
#
#     for k in X:
#
#         emb_list.append(embeddings[k])
#
#     emb_list = np.array(emb_list)
#
#
#
#     model = TSNE(n_components=2)
#
#     node_pos = model.fit_transform(emb_list)
#
#
#
#     color_idx = {}
#
#     for i in range(len(X)):
#
#         color_idx.setdefault(Y[i][0], [])
#
#         color_idx[Y[i][0]].append(i)
#
#
#
#     for c, idx in color_idx.items():
#
#         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
#
#     plt.legend()
#
#     plt.show()

if __name__ == "__main__":
    G = nx.read_edgelist("../data/drug/edge.txt", create_using=nx.DiGraph(), nodetype=None,
                         data=[('weight', int)])

    model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )
    model.train()
    embeddings = model.get_embeddings()

    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
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
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    print(classification_report(Y_test, Y_pred))
    print("-------------------------")
    mcc = matthews_corrcoef(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print("MCC:", mcc)
    print("AUC:", auc)