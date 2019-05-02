import numpy as np
import os
import sys

from sklearn import svm
from sklearn.decomposition import PCA

def LoadDataSet(list_path):
    fin = open(list_path, 'r')
    lines = fin.readlines()
    data = []
    label = []
    for line in lines:
        line = line.strip()
        line_list = line.split('\t')
        label.append(int(line_list[0]))
        feature = line_list[1].split(' ')
        feature = list(map(lambda x:float(x), feature))
        data.append(feature)
    data = np.array(data)
    label = np.array(label)
    return data, label

def main():
    BTW_clusters = [32, 64, 128, 256, 512]
    VLAD_clusters = [8, 16, 32, 64, 128]
    FisherVector_clusters = [[4, 32], [6, 64], [8, 128], [8, 32]]
    for feature_dims in VLAD_clusters:
        train_list_path = 'VLAD-' + str(feature_dims) + '-c50-features-train.txt'
        test_list_path = 'VLAD-' + str(feature_dims) + '-c50-features-test.txt'

        print('Working on cluster num ', feature_dims)

        train_data, train_label = LoadDataSet(train_list_path)
        test_data, test_label = LoadDataSet(test_list_path)

        pca_class = PCA(n_components=feature_dims)
        pca_model = pca_class.fit(train_data)
        train_data = pca_model.transform(train_data)
        test_data = pca_model.transform(test_data)

        print('train_data shape', train_data.shape)
        print('train_label shape', train_label.shape)
        print('test_data shape', test_data.shape)
        print('test_label shape', test_label.shape)

        classifier = svm.SVC(C=1.0, kernel='rbf', gamma='scale', decision_function_shape='ovr')
        print('Start Training...')
        classifier.fit(train_data, train_label)
        print('Finish Training!')
        #print('Train Accuracy', classifier.score(train_data, train_label))
        print('Accuracy', classifier.score(test_data, test_label))

if __name__ == '__main__':
    main()