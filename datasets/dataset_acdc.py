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

SPACING_KEYS = (
    "voxelspacing_zyx",
    "spacing_zyx",
    "voxelspacing",
    "spacing",
    "spacing_mm",
    "pixdim",
    "zooms",
)
SPACING_ZYX_KEYS = {"voxelspacing_zyx", "spacing_zyx"}

class ACDC_Dataset(Dataset):
    def __init__(self, base_dir=None, split='train', list_dir=None, transform=None, fold_id=0):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        train_ids, val_ids, test_ids = self._get_ids()
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

    # def _get_ids(self, fold_id=0):
    #     """Return patient ID splits for training, validation, and testing."""
    #     all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
    #     fold_id = int(fold_id)
    #     if fold_id < 0 or fold_id >= 5:
    #         raise ValueError("fold_id must be between 0 and 4")

    #     fold_size = len(all_cases_set) // 5
    #     folds = [
    #         all_cases_set[i * fold_size:(i + 1) * fold_size]
    #         for i in range(5)
    #     ]
    #     validation_fold_id = (fold_id + 1) % 5

    #     testing_set = folds[fold_id]
    #     validation_set = folds[validation_fold_id][:10]
    #     training_set = [
    #         case
    #         for case in all_cases_set
    #         if case not in testing_set and case not in validation_set
    #     ]

    #     training_ids = set(training_set)
    #     validation_ids = set(validation_set)
    #     testing_ids = set(testing_set)
    #     assert len(training_set) == 70
    #     assert len(validation_set) == 10
    #     assert len(testing_set) == 20
    #     assert not training_ids & validation_ids
    #     assert not training_ids & testing_ids
    #     assert not validation_ids & testing_ids
    #     assert len(training_ids | validation_ids | testing_ids) == 100

    #     return [training_set, validation_set, testing_set]
    
    def _get_ids(self):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        testing_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
        validation_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
        training_set = [i for i in all_cases_set if i not in testing_set+validation_set]

        return [training_set, validation_set, testing_set]

    def _normalize_spacing_zyx(self, spacing, key, case):
        spacing = np.asarray(spacing, dtype=np.float32).reshape(-1)
        if key == "pixdim" and spacing.size >= 4:
            spacing = spacing[1:4]
        else:
            spacing = spacing[:3]
        if spacing.size != 3:
            raise ValueError("ACDC case {} spacing key {} must contain 3 spatial values".format(case, key))
        if key not in SPACING_ZYX_KEYS:
            spacing = spacing[::-1]
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError("ACDC case {} spacing key {} has invalid values {}".format(case, key, spacing.tolist()))
        return spacing.astype(np.float32)

    def _spacing_from_h5(self, h5f, case):
        for key in SPACING_KEYS:
            if key in h5f.attrs:
                return self._normalize_spacing_zyx(h5f.attrs[key], key, case)
            if key in h5f:
                return self._normalize_spacing_zyx(h5f[key][()], key, case)
            for dataset_key in ("image", "label"):
                if dataset_key in h5f and key in h5f[dataset_key].attrs:
                    return self._normalize_spacing_zyx(h5f[dataset_key].attrs[key], key, case)
        return None

    def _nifti_spacing_zyx(self, path, case):
        try:
            import nibabel as nib

            return self._normalize_spacing_zyx(nib.load(path).header.get_zooms()[:3], "zooms", case)
        except ImportError:
            pass

        try:
            import SimpleITK as sitk

            return self._normalize_spacing_zyx(sitk.ReadImage(path).GetSpacing()[:3], "spacing", case)
        except ImportError:
            pass
        return None

    def _original_nifti_candidates(self, case):
        stem = case.replace(".h5", "")
        patient_id = stem.split("_")[0]
        directories = [
            self._base_dir,
            os.path.join(self._base_dir, "ACDC_training_volumes"),
            os.path.join(self._base_dir, patient_id),
            os.path.join(self._base_dir, "training", patient_id),
            os.path.join(self._base_dir, "database", "training", patient_id),
            os.path.join(self._base_dir, "ACDC_training", patient_id),
        ]
        candidates = []
        for directory in directories:
            candidates.append(os.path.join(directory, stem + ".nii.gz"))
            candidates.append(os.path.join(directory, stem + ".nii"))
        return candidates

    def _volume_spacing_zyx(self, h5f, case):
        spacing = self._spacing_from_h5(h5f, case)
        if spacing is not None:
            return spacing
        for path in self._original_nifti_candidates(case):
            if os.path.exists(path):
                spacing = self._nifti_spacing_zyx(path, case)
                if spacing is not None:
                    return spacing
        return None

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        if self.split == "train":
            with h5py.File(self._base_dir + "/ACDC_training_slices/{}".format(case), 'r') as h5f:
                image = h5f['image'][:]
                label = h5f['label'][:]  # fix sup_type to label
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            with h5py.File(self._base_dir + "/ACDC_training_volumes/{}".format(case), 'r') as h5f:
                image = h5f['image'][:]
                label = h5f['label'][:]
                voxelspacing_zyx = self._volume_spacing_zyx(h5f, case)
            sample = {'image': image, 'label': label}
            if voxelspacing_zyx is not None:
                sample['voxelspacing_zyx'] = voxelspacing_zyx
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
        if augment > 0.6:
            image, label = random_rot_flip(image, label)
        elif augment > 0.35:
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
