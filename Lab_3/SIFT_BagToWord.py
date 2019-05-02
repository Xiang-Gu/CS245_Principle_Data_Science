import cv2
import numpy as np
import os
import random
from scipy.cluster.vq import *
from sklearn import preprocessing

sift_det = cv2.xfeatures2d.SIFT_create()

resize_height = 224
resize_width = 224

train_list_path = 'train_list.txt'
test_list_path = 'test_list.txt'

cluster_nums = [32, 64, 128, 256, 512]
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

for cluster_num in cluster_nums:
    print('For cluster num', cluster_num)
    print('Start K-means stage')
    # Perform k-means clustering
    voc, variance = kmeans(descriptors, cluster_num, 1)
    print('Finish K-means stage')

    # Calculate the histogram of features
    train_im_features = np.zeros((len(train_image_paths), cluster_num), 'float32')
    for i in xrange(len(train_image_paths)):
        words, distance = vq(train_des_list[i][1], voc)
        for w in words:
            train_im_features[i][w] += 1
    print('Finish Train BTW stage')

    test_im_features = np.zeros((len(test_image_paths), cluster_num), 'float32')
    for i in xrange(len(test_image_paths)):
        words, distance = vq(test_des_list[i][1], voc)
        for w in words:
            test_im_features[i][w] += 1
    print('Finish Test BTW stage')  

    # Perform L2 normalization
    train_im_features = preprocessing.normalize(train_im_features, norm='l2')
    test_im_features = preprocessing.normalize(test_im_features, norm='l2')

    print('train_im_features shape', train_im_features.shape)
    print('test_im_features shape', test_im_features.shape)

    train_fout_feature = open('BTW-' + str(cluster_num) + '-features-train.txt', 'w')
    for idx in xrange(len(train_image_paths)):
        train_fout_feature.write(train_image_labels[idx] + '\t')
        for j in xrange(cluster_num):
            train_fout_feature.write(str(train_im_features[idx][j]) + ' ')
        train_fout_feature.write('\n')
    train_fout_feature.close()

    test_fout_feature = open('BTW-' + str(cluster_num) + '-features-test.txt', 'w')
    for idx in xrange(len(test_image_paths)):
        test_fout_feature.write(test_image_labels[idx] + '\t')
        for j in xrange(cluster_num):
            test_fout_feature.write(str(test_im_features[idx][j]) + ' ')
        test_fout_feature.write('\n')
    test_fout_feature.close()
