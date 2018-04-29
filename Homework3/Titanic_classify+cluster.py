import pandas as pd
from sklearn import svm, tree
import pydotplus
from matplotlib import pylab as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
import numpy as np
from sklearn.manifold import TSNE


import os
os.environ["PATH"] += os.pathsep + 'D:\software\graphviz-2.38\\bin'


# 分类
def decision_tree(x_train, x_test, y_train, y_test, create_graph):

    dec_classifier = tree.DecisionTreeClassifier(max_depth=7)
    dec_classifier.fit(x_train, y_train)


    print("决策树分类准确率:" + str(dec_classifier.score(x_test, y_test)))

    if create_graph == True:
        dot_data = tree.export_graphviz(dec_classifier, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png("decision_tree.png")
    return dec_classifier


def svm_classify(x_train, x_test, y_train, y_test):

    svm_classifier = svm.SVC(C=0.5, kernel='rbf', gamma=0.5)
    svm_classifier.fit(x_train, y_train)
    print("svm分类准确率:" + str(svm_classifier.score(x_test, y_test)))
    return svm_classifier


def visualization(classifiers, x_data, y_data, title):
    x1_min, x1_max = x_data[:, 0].min()-1, x_data[:, 0].max()+1
    x2_min, x2_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    t1 = np.linspace(x1_min, x1_max, 100)
    t2 = np.linspace(x2_min, x2_max, 100)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.flat, x2.flat), axis=1)

    y_hat = classifiers.predict(x_show)  # 预测
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同

    light = mpl.colors.ListedColormap(['y', 'skyblue'])
    dark = mpl.colors.ListedColormap(['yellow', 'blue'])

    plt.figure(facecolor='w')
    plt.title(title)
    plt.pcolormesh(x1, x2, y_hat, cmap=light)
    plt.scatter(x_data[:, 0], x_data[:, 1], s=30, c=y_data, edgecolors='k', cmap=dark)
    plt.savefig(title+".png")
    plt.show()
    plt.close()


# 聚类
def cluster(data, columns):
    data_copy = data.copy()
    for column in columns:
        data_copy[column] = data_copy[column].replace(generate_dict(data_copy[column]))
    y_data = data_copy['Survived'].as_matrix()
    x_data = data_copy[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].as_matrix()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=1, train_size=0.9, test_size=0.1)
    X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(x_data)
    # k-means 聚类
    k_means = KMeans(init="k-means++", n_clusters=2, random_state=28)
    k_means.fit(X_tsne)
    y_km = k_means.predict(X_tsne)
    km_center = k_means.cluster_centers_
    # 密度聚类
    y_db = DBSCAN(eps=0.4, min_samples=2).fit_predict(X_tsne)
    cm_light = mpl.colors.ListedColormap(['yellow', 'skyblue'])
    cm2 = mpl.colors.ListedColormap(['y', 'blue'])
    plt.title("original scatter graph")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=30, c=y_data, edgecolors='k', cmap=cm_light)
    plt.savefig("原始图" + ".png")
    plt.show()
    plt.title("k-means")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=30, c=y_km, edgecolors='k', cmap=cm_light)
    plt.scatter(km_center[:, 0], km_center[:, 1], c=range(2), s=60, cmap=cm2,
                edgecolors='none')
    plt.savefig("km" + ".png")
    plt.show()
    plt.title("DBSCAN")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=30, c=y_db, edgecolors='k')
    plt.savefig("DBSCAN" + ".png")
    plt.show()
    plt.close()


def generate_dict(column):

    value = list(set(column))
    return {value[i]: i for i in range(value.__len__())}


if __name__ == '__main__':
    data = pd.read_csv(r'E:\datamining\homework3\train.csv', low_memory=False)
    columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    data_copy = data.copy()
    for column in columns:
        data_copy[column] = data_copy[column].replace(generate_dict(data_copy[column]))
    y_data = data_copy['Survived'].as_matrix()
    x_data = data_copy[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Cabin']].as_matrix()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=1, train_size=0.9, test_size=0.1)
    decision_tree(x_train, x_test, y_train, y_test, True)
    svm_classify(x_train, x_test, y_train, y_test)
    X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(x_data)
    # X_lda = LinearDiscriminantAnalysis(n_components=2).fit(x_data, y_data).transform(x_data)
    # X_pca = PCA().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(X_tsne, y_data, random_state=1, train_size=0.9, test_size=0.1)
    model1 = decision_tree(x_train, x_test, y_train, y_test, False)
    visualization(model1, X_tsne, y_data, "decision_tree_classify")
    model2 = svm_classify(x_train, x_test, y_train, y_test)
    visualization(model2, X_tsne, y_data, "svm_classify")
    #聚类
    cluster(data, columns)




