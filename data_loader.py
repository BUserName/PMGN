from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import random
import os
import numpy as np
import scipy.io as sio

class Base_Dataset(data.Dataset):
    def __init__(self, root, partition, target_ratio=0):
        super(Base_Dataset, self).__init__()
        # set dataset info
        self.root = root
        self.partition = partition
        self.target_ratio = target_ratio
        # self.target_ratio=0 no mixup

    def __len__(self):

        if self.partition == 'train':
            return int(min(sum(self.alpha), len(self.target_image)) / (self.num_class ))
        elif self.partition == 'test':
            return int(len(self.target_image) / (self.num_class))

    def __getitem__(self, item):

        src_image_data = [] # Source Image feature
        src_label_data = [] # Source Label

        tar_image_data = [] # Target Image
        tar_label_data = [] # Image Pseudo-label /
                            # If a sample is not yet labeled, the label equals to the total num of class

        target_real_label = [] # Target Ground Truth
        class_index_target = []

        known_label_mask = []
        ST_split = [] # Mask of targets to be evaluated
        # select index for support class

        class_index_source = list(set(range(self.num_class)))
        random.shuffle(class_index_source)
        unlabel_idx = np.where(self.target_known_mask < 1)[0]
        unlabel_tar_img = torch.FloatTensor(self.target_image)[unlabel_idx]
        unlabel_tar_label = torch.LongTensor(self.target_label)[unlabel_idx]
        for classes in class_index_source:
            # select support samples from source domain or target domain
            image = torch.FloatTensor(random.choice(self.source_image[classes]))
            src_image_data.append(image)
            src_label_data.append(classes)
            # known_label_mask.append(1)
            ST_split.append(0)
            # target_real_label.append(classes)


        # adding target samples
        if self.partition == 'train':
            num_support = int(self.num_class * self.target_ratio / self.num_class)# 0.3
            for i in range(self.num_class - num_support):
                if self.target_ratio > 0:
                    index = random.choice(list(range(len(self.label_flag))))
                else:
                    index = random.choice(list(range(len(self.target_image))))
                target_image = torch.FloatTensor(self.target_image[index])

                tar_image_data.append(target_image)
                tar_label_data.append(self.num_class) # label_flag
                target_real_label.append(self.target_label[index])
                known_label_mask.append(0)
                ST_split.append(1)
            for i in range(num_support):
                index = random.choice(list(range(self.reference_num)))
                target_image = torch.FloatTensor(self.target_image[index])
                tar_image_data.append(target_image)
                tar_label_data.append(self.target_label[index])
                target_real_label.append(self.target_label[index])
                known_label_mask.append(1)
                ST_split.append(1)
            # shuffle the order
            tar_zipped = list(zip(tar_image_data, tar_label_data, target_real_label, known_label_mask))
            np.random.shuffle(tar_zipped)
            tar_image_data, tar_label_data, target_real_label, known_label_mask = zip(*tar_zipped)

        elif self.partition == 'test':
            for i in range(self.num_class):
                target_image = unlabel_tar_img[(item) * (self.num_class - 1) + i]

                tar_image_data.append(target_image)
                tar_label_data.append(self.num_class)
                target_real_label.append(unlabel_tar_label[(item) * (self.num_class - 1) + i])
                known_label_mask.append(0)
                ST_split.append(1)
        source_known = np.ones_like(known_label_mask)
        known_label_mask = np.concatenate([source_known, known_label_mask])
        src_image_data = torch.stack(src_image_data)
        src_label_data = torch.LongTensor(src_label_data)
        tar_image_data = torch.stack(tar_image_data)
        tar_label_data = torch.LongTensor(tar_label_data)
        real_label_data = torch.tensor(target_real_label)
        known_label_mask = torch.tensor(known_label_mask)
        ST_split = torch.tensor(ST_split)
        return [src_image_data, tar_image_data], [src_label_data, tar_label_data], real_label_data, known_label_mask, ST_split

    def load_dataset(self, known_num):
        source_image_list = {key: [] for key in range(self.num_class)}
        target_image_list = []
        target_label_list = []

        src_mat = sio.loadmat(self.source_path)
        tar_mat = sio.loadmat(self.target_path)


        for ind in range(src_mat['data'].shape[1]):
            img_fea = src_mat['data'][:,ind]
            label = src_mat['label'][0][ind]
            source_image_list[int(label-1)].append(img_fea)

        for ind in range(tar_mat['data'].shape[1]):
            img_fea = tar_mat['data'][:,ind]
            label = tar_mat['label'][0][ind]
            # target_image_list[int(label)].append(image_dir)
            target_image_list.append(img_fea)
            target_label_list.append(int(label-1))
        self.target_known_mask = np.zeros(len(target_image_list))
        self.target_known_mask[0:known_num] = 1
        return source_image_list, target_image_list, target_label_list


class Office_Dataset(Base_Dataset):

    def __init__(self, root, partition, label_flag=None, source='A', target='W', target_ratio=0.0):
        super(Office_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        self.source_path = source
        self.target_path = target
        self.class_name = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        self.num_class = len(self.class_name)
        self.target_ratio = self.num_class - target_ratio
        self.reference_num = self.num_class * 3
        self.source_image, self.target_image, self.target_label = self.load_dataset(self.reference_num)
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag
        if self.label_flag is None:
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class
            self.label_flag[np.where(self.target_known_mask==1)] = torch.FloatTensor(self.target_label)[np.where(self.target_known_mask==1)]

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])

        self.alpha_value = self.alpha
        self.alpha_value = np.array(self.alpha_value)
        self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
        self.alpha_value = torch.tensor(self.alpha_value).float().cuda()

class NUSIMG_Dataset(Base_Dataset):

    def __init__(self, root, partition, label_flag=None, source='A', target='W', target_ratio=0.0):
        super(NUSIMG_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        self.source_path = source
        self.target_path = target
        self.class_name = ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.num_class = len(self.class_name)
        self.target_ratio = self.num_class - target_ratio
        self.reference_num = self.num_class * 3
        self.source_image, self.target_image, self.target_label = self.load_dataset(self.reference_num)
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag
        if self.label_flag is None:
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class
            self.label_flag[np.where(self.target_known_mask == 1)] = torch.FloatTensor(self.target_label)[
                np.where(self.target_known_mask == 1)]

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])

        self.alpha_value = self.alpha
        self.alpha_value = np.array(self.alpha_value)
        self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
        self.alpha_value = torch.tensor(self.alpha_value).float().cuda()


class MRC_Dataset(Base_Dataset):

    def __init__(self, root, partition, label_flag=None, source='A', target='W', target_ratio=0.0):
        super(MRC_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        self.source_path = source
        self.target_path = target
        self.class_name = ["1", "2", "3", "4", "5", "6"]
        self.num_class = len(self.class_name)
        self.target_ratio = self.num_class - target_ratio
        self.reference_num = self.num_class * 10
        # self.idx = idx
        self.source_image, self.target_image, self.target_label = self.load_dataset(self.reference_num)
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag
        if self.label_flag is None:
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class
            self.label_flag[np.where(self.target_known_mask == 1)] = torch.FloatTensor(self.target_label)[
                np.where(self.target_known_mask == 1)]

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])

        self.alpha_value = self.alpha
        self.alpha_value = np.array(self.alpha_value)
        self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
        self.alpha_value = torch.tensor(self.alpha_value).float().cuda()

    def load_dataset(self, known_num):
        source_image_list = {key: [] for key in range(self.num_class)}
        target_image_list = []
        target_label_list = []


        src_mat = sio.loadmat(self.source_path)
        tar_mat = sio.loadmat(self.target_path)
        # tar_mat_t = sio.loadmat('/home/data1/mrc/test_sp2.mat')
        # tar_idx = np.random.permutation(tar_mat['testing_features'][2,0].shape[0])[:440]
        for ind in range(src_mat['source_features'][3,0].shape[0]):
            img_fea = src_mat['source_features'][3,0][ind]
            label = src_mat['source_labels'][3,0][ind]
            source_image_list[int(label-1)].append(img_fea)

        for ind in range(tar_mat['training_features'][3,0].shape[0]):
            img_fea = tar_mat['training_features'][3,0][ind]
            label = tar_mat['training_labels'][3,0][ind]
            # target_image_list[int(label)].append(image_dir)
            target_image_list.append(img_fea)
            target_label_list.append(int(label-1))

        for ind in range(tar_mat['testing_features'][3,0].shape[0]):
            img_fea = tar_mat['testing_features'][3,0][ind]
            label = tar_mat['testing_labels'][3,0][ind]
            # target_image_list[int(label)].append(image_dir)
            target_image_list.append(img_fea)
            target_label_list.append(int(label-1))


        self.target_known_mask = np.zeros(len(target_image_list))
        self.target_known_mask[0:known_num] = 1
        return source_image_list, target_image_list, target_label_list





