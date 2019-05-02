import cv2
import numpy as np
import os
import random
from scipy.cluster.vq import *
from sklearn import preprocessing
from sklearn.decomposition import PCA

from FisherVector import generate_gmm, fisher_vector

sift_det = cv2.xfeatures2d.SIFT_create()

resize_height = 224
resize_width = 224

train_list_path = 'train_c50_list.txt'
test_list_path = 'test_c50_list.txt'

cluster_nums = [4, 6, 8]
pca_dims = [[32], [64], [128, 32]]
print('cluster_nums', cluster_nums)

#Train Data vvv=======================================================================================================
fin_train = open(train_list_path, 'r')
train_lines = fin_train.readlines()

train_des_list = []
train_image_paths = []
train_image_labels = []

descriptors = None
tmp_des = None

print('Start Reading Train Data')
count = 0
# Read images and use SIFT to get keypoints and descriptors
for line in train_lines:
    if line == '':
        continue
    line = line.strip()
    line_list = line.split('\t')
    file_path = line_list[0]
    label = line_list[1]
    img = cv2.imread(file_path)
    img = cv2.resize(img, (resize_width, resize_height))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift_det.detectAndCompute(gray, None)
    if random.random() < 0.1:
        if tmp_des is None:
            tmp_des = des
        else:
            tmp_des = np.vstack((tmp_des, des))
    train_des_list.append((file_path, des))
    train_image_paths.append(file_path)
    train_image_labels.append(label)
    count += 1
    if count % 500 == 0:
        print('Running on', count)
        if descriptors is None:
            descriptors = tmp_des
        else:
            descriptors = np.vstack((descriptors, tmp_des))
        tmp_des = None
fin_train.close()

if tmp_des is not None:
    if descriptors is None:
        descriptors = tmp_des
    else:
        descriptors = np.vstack((descriptors, tmp_des))

print('Finish Reading Train Data and SIFT Stage')
#Train Data ^^^=======================================================================================================

#Test Data vvv========================================================================================================
fin_test = open(test_list_path, 'r')
test_lines = fin_test.readlines()

print('Start Reading Test Data')

test_des_list = []
test_image_paths = []
test_image_labels = []

count = 0
# Read images and use SIFT to get keypoints and descriptors
for line in test_lines:
    if line == '':
        continue
    line = line.strip()
    line_list = line.split('\t')
    file_path = line_list[0]
    label = line_list[1]
    img = cv2.imread(file_path)
    img = cv2.resize(img, (resize_width, resize_height))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift_det.detectAndCompute(gray, None)
    test_des_list.append((file_path, des))
    test_image_paths.append(file_path)
    test_image_labels.append(label)
    count += 1
    if count % 500 == 0:
        print('Running on', count)
fin_test.close()
print('Finish Reading Test Data and SIFT Stage')
#Test Data ^^^========================================================================================================

print('descriptor shape', descriptors.shape)

for c_idx in xrange(len(cluster_nums)):
    cluster_num = cluster_nums[c_idx]
    print('For cluster num', cluster_num)
    print('Start GMM stage')
    # Perform GMM clustering
    gmm_model = generate_gmm(descriptors, cluster_num)
    print('Finish GMM stage')

    # Calculate the histogram of features
    train_im_features = []
    for i in xrange(len(train_image_paths)):
        feat = fisher_vector(train_des_list[i][1], *gmm_model)
        train_im_features.append(feat)
    print('Finish Train FisherVector stage')

    test_im_features = []
    for i in xrange(len(test_image_paths)):
        feat = fisher_vector(test_des_list[i][1], *gmm_model)
        test_im_features.append(feat)
    print('Finish Test FisherVector stage')

    train_im_features = np.array(train_im_features)
    test_im_features = np.array(test_im_features)

    train_im_features = train_im_features.reshape(train_im_features.shape[0], -1)
    test_im_features = test_im_features.reshape(test_im_features.shape[0], -1)

    print('train_im_features before PCA shape', train_im_features.shape)
    print('test_im_features before PCA shape', test_im_features.shape)

    dims = pca_dims[c_idx]
    for now_pca_dim in dims:
        if train_im_features.shape[1] > now_pca_dim:
            pca_class = PCA(n_components=now_pca_dim)
            pca_model = pca_class.fit(train_im_features)
            train_im_features = pca_model.transform(train_im_features)
            test_im_features = pca_model.transform(test_im_features)

        # Perform L2 normalization
        #train_im_features = preprocessing.normalize(train_im_features, norm='l2')
        #test_im_features = preprocessing.normalize(test_im_features, norm='l2')

        print('train_im_features shape', train_im_features.shape)
        print('test_im_features shape', test_im_features.shape)

        train_fout_feature = open('FisherVector-' + str(cluster_num) + 'c-' + str(train_im_features.shape[1]) + 'd-features-train.txt', 'w')
        for idx in xrange(len(train_image_paths)):
            train_fout_feature.write(train_image_labels[idx] + '\t')
            for j in xrange(train_im_features.shape[1]):
                train_fout_feature.write(str(train_im_features[idx][j]) + ' ')
            train_fout_feature.write('\n')
        train_fout_feature.close()

        test_fout_feature = open('FisherVector-' + str(cluster_num) + 'c-' + str(test_im_features.shape[1]) + 'd-features-test.txt', 'w')
        for idx in xrange(len(test_image_paths)):
            test_fout_feature.write(test_image_labels[idx] + '\t')
            for j in xrange(test_im_features.shape[1]):
                test_fout_feature.write(str(test_im_features[idx][j]) + ' ')
            test_fout_feature.write('\n')
        test_fout_feature.close()
