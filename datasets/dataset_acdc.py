import itertools
import os
import random
import re
# from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
# from skimage import io
# import cv2

class ACDC_Dataset(Dataset):
    def __init__(self, base_dir=None, split='train', list_dir=None, transform=None, fold_id=0):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        train_ids, val_ids, test_ids = self._get_ids(fold_id=fold_id)
        if self.split.find('train') != -1:
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split.find('val') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        elif self.split.find('test') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_ids(self, fold_id=0):
        """Return patient ID splits for training, validation, and testing.

        The dataset is divided into 5 equal folds. The requested fold_id is used
        as the test set, the next fold is used for validation, and the remaining
        folds are combined for training.

        Args:
            fold_id (int): index of the fold to use as the test set (0-4).

        Returns:
            list: [training_set, validation_set, testing_set] where each element
            is a list of patient IDs.
        """
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        # fold_id = int(fold_id)
        fold_size = len(all_cases_set) // 5
        folds = [
            all_cases_set[i * fold_size:(i + 1) * fold_size]
            for i in range(5)
        ]
        validation_fold_id = (fold_id + 1) % 5
        testing_set = folds[fold_id]
        validation_set = folds[validation_fold_id]
        training_set = [
            case
            for i, fold in enumerate(folds)
            if i not in (fold_id, validation_fold_id)
            for case in fold
        ]
        # else:
        #     testing_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
        #     validation_set = ["patient{:0>3}".format(i) for i in range(21, 31)]
        #     training_set = [i for i in all_cases_set if i not in testing_set+validation_set]

        return [training_set, validation_set, testing_set]
    
    # def _get_ids(self, seed=1234):
    #     all_cases = [f"patient{i:03d}" for i in range(1, 101)]
    #     rng = random.Random(seed)
    #     rng.shuffle(all_cases)

    #     test_ids = all_cases[:20]
    #     val_ids  = all_cases[20:30]
    #     train_ids = all_cases[30:]

    #     return [train_ids, val_ids, test_ids]


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/ACDC_training_slices/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]  # fix sup_type to label
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            h5f = h5py.File(self._base_dir + "/ACDC_training_volumes/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
        sample["idx"] = idx
        sample['case_name'] = case.replace('.h5', '')
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator4ACDC(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        augment = random.random()
        if augment > 0.5:
            image, label = random_rot_flip(image, label)
        elif augment > 0.25:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # the default is 0
            label = zoom( label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        assert (image.shape[0] == self.output_size[0]) and (image.shape[1] == self.output_size[1])
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
