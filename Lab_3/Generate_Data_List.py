import os
import random
import sys

dataset_ratio = 1.0
train_image_ratio = 0.6

dataset_path = 'D:\BaiduNetdiskDownload\Animals_with_Attributes2\JPEGImages'

category_files = os.listdir(dataset_path)

category_num = len(category_files)

random.shuffle(category_files)
category_num = int(category_num * dataset_ratio)
category_files = category_files[:category_num]

print('category_num: ', category_num)

train_images = []
train_labels = []

test_images = []
test_labels = []

for idx in xrange(category_num):
    now_file = category_files[idx]
    now_file_path = os.path.join(dataset_path, now_file)
    now_images = os.listdir(now_file_path)
    now_image_num = len(now_images)
    train_num = int(now_image_num * train_image_ratio)
    random.shuffle(now_images)
    train_images.extend([os.path.join(now_file_path, x) for x in now_images[:train_num]])
    train_labels.extend([idx] * train_num)

    test_images.extend([os.path.join(now_file_path, x) for x in now_images[train_num:]])
    test_labels.extend([idx] * (now_image_num - train_num))
    print(now_file, now_image_num, 'train: ', train_num, 'test: ', (now_image_num - train_num))

assert len(train_images) == len(train_labels)
assert len(test_images) == len(test_labels)

print('total: ', len(train_images) + len(test_images), 'train: ', len(train_images), 'test: ', len(test_images))

fout_train = open('train_list.txt', 'w')
for idx in xrange(len(train_images)):
    fout_train.write(train_images[idx] + '\t' + str(train_labels[idx]) + '\n')
fout_train.close()

fout_test = open('test_list.txt', 'w')
for idx in xrange(len(test_images)):
    fout_test.write(test_images[idx] + '\t' + str(test_labels[idx]) + '\n')
fout_test.close()

