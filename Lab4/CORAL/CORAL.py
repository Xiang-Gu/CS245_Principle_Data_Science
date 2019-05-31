# encoding=utf-8
"""
    Created on 16:31 2018/11/13 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
import pandas as pd
from time import time
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.decomposition import PCA



WHERTHER_PCA = True


class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        t0=time()
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral)
        print("fit done in ",time()-t0)
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        global names
        names = [
                 # "Nearest Neighbors",
                 # "Linear SVM",
                 "RBF SVM",
                 # "Gaussian Process",
                 # "Decision Tree",
                 # "Random Forest",
                 "Extra Tree",
                 # "Neural Net",
                 # "AdaBoost",
                 # "Naive Bayes",
                 # "QDA"
                 ]

        classifiers = [
            # KNeighborsClassifier(1),
            # SVC(kernel="linear", C=2.5),
            SVC(gamma=0.2, C=0.25),
            # GaussianProcessClassifier(1.0 * RBF(1.0)),
            # DecisionTreeClassifier(max_depth=5),
            # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            ExtraTreesClassifier(n_estimators=100, max_depth=None),
            # MLPClassifier(alpha=1, max_iter=2000,hidden_layer_sizes=(200,100)),
            ]
        acc = []
        for name, clf in zip(names, classifiers):
            t0 = time()
            print('begin %s fit' % name)
            if name == 'RBF SVM':
                params = {"C": [0.1, 1, 10], "gamma": [0.1, 0.01, 0.001]}
                clf = GridSearchCV(clf, params)
                # .fit(Xs_new, Ys.ravel())
            clf.fit(Xs_new, Ys.ravel())
            # print('clf fit done')
            y_pred = clf.predict(Xt)
            acc.append(sklearn.metrics.accuracy_score(Yt, y_pred))
            print("clf done in %0.3fs" %(time()-t0))
        # clf = KNeighborsClassifier(n_neighbors=1)
        # clf.fit(Xs_new, Ys.ravel())
        # y_pred = clf.predict(Xt)
        # acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred


if __name__ == '__main__':
    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    domains = ['Art_Art.csv',"Clipart_Clipart.csv","Product_Product.csv"]
    t_domains = ["Art_RealWorld.csv","Clipart_RealWorld.csv","Product_RealWorld.csv"]
    datapath = "../data/Office-Home_resnet50/"
    for i in range(len(domains)):
        for j in range((len(domains))):
            if i == j:
                print("source:",domains[i])
                print("target",t_domains[j])
                src, tar = datapath + domains[i], datapath + t_domains[j]
                # src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Source = pd.read_csv(src,header=None)
                Target = pd.read_csv(tar,header=None)
                Ys = Source[2048]
                Xs = Source.iloc[:, 0:2048]
                Yt = Target[2048]
                Xt = Target.iloc[:, 0:2048]
                # Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                if WHERTHER_PCA:
                    t0=time()
                    pca = PCA(n_components=0.95,svd_solver='full').fit(Xs)
                    # print("done in %0.3fs" % (time() - t0))
                    # t0 = time()
                    Xs = pca.transform(Xs)
                    Xt = pca.transform(Xt)
                    print(Xs.shape,Xs.shape)
                    # print("done in %0.3fs" % (time() - t0))
                # src, tar = datapath + domains[i], '/Users/chenchacha/tra .nsferlearning/code/traditional/data/' + t_domains[j]
                # src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                # Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                coral = CORAL()
                accs, _ = coral.fit_predict(Xs, Ys, Xt, Yt)
                # print(acc)
                for name,acc in zip(names,accs):
                    print(name,acc)
