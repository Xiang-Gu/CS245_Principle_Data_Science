# encoding=utf-8
"""
    Created on 21:29 2018/11/12 
    @author: Jindong Wang
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
from sklearn.decomposition import PCA
from time import time

Classifier = ['KNN','SVC','DEEP'][1]
WHERTHER_PCA = True

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        print("begin TCA fit")
        t0=time()
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        print("fit done in %0.3fs" % (time() - t0))
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        # if Classifier=='KNN':
        #     clf = KNeighborsClassifier(n_neighbors=1)
        # elif Classifier=='SVC':
        #     clf = SVC(kernel='linear',C=2.5)
        global names
        names = ["Nearest Neighbors",
                 "Linear SVM",
                 "RBF SVM",
                 # "Gaussian Process",
                 # "Decision Tree",
                 # "Random Forest",
                 "Extra Tree",
                 "Neural Net",
                 # "AdaBoost",
                 # "Naive Bayes",
                 # "QDA"
                 ]

        classifiers = [
            KNeighborsClassifier(1),
            SVC(kernel="linear", C=2.5),
            SVC(gamma=2, C=2.5),
            # GaussianProcessClassifier(1.0 * RBF(1.0)),
            # DecisionTreeClassifier(max_depth=5),
            # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split = 2, random_state = 0),
            MLPClassifier(alpha=1, max_iter=2000),
            ]
        # names = names[:1]
        # classifiers = classifiers[:1]
        acc = []
        for name, clf in zip(names, classifiers):
            t0 = time()
            print('begin %s fit' % name)
            clf.fit(Xs_new, Ys.ravel())
            # print('clf fit done')
            y_pred = clf.predict(Xt_new)
            acc.append(sklearn.metrics.accuracy_score(Yt, y_pred))
            print("clf done in %0.3fs" %(time()-t0))
        return acc, y_pred


if __name__ == '__main__':
    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    domains = ['Art_Art.csv',"Clipart_Clipart.csv","Product_Product.csv"]
    t_domains = ["Art_RealWorld.csv","Clipart_RealWorld.csv","Product_RealWorld.csv"]
    datapath = "../data/Office-Home_resnet50/"
    for i in range(len(domains)):
        for j in range(len(domains)):
            if i == j:
                print("source:", domains[i])
                print("target", t_domains[j])
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
                    pca = PCA(n_components=0.9,svd_solver='full').fit(Xs)
                    print("done in %0.3fs" % (time() - t0))
                    t0 = time()
                    Xs_new = pca.transform(Xs)
                    Xt_new = pca.transform(Xt)
                    print(Xs.shape,Xs_new.shape)
                    print("done in %0.3fs" % (time() - t0))

                tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
                accs, ypre = tca.fit_predict(Xs_new, Ys, Xt_new, Yt)
                for name,acc in zip(names,accs):
                    print(name,acc)

                # print(acc)
                # It should print 0.910828025477707
