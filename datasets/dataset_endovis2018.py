import json
import os
import random

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


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


class RandomGenerator4EndoVis2018(object):
    def __init__(self, output_size, augment=True):
        self.output_size = output_size
        self.augment = augment

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.augment and random.random() > 0.6:
            image, label = random_rot_flip(image, label)
        elif self.augment and random.random() > 0.35:
            image, label = random_rotate(image, label)

        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            image = zoom(
                image,
                (self.output_size[0] / h, self.output_size[1] / w, 1),
                order=3,
            )
            label = zoom(
                label,
                (self.output_size[0] / h, self.output_size[1] / w),
                order=0,
            )

        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class EndoVis2018Dataset(Dataset):
    def __init__(
        self,
        base_dir,
        split='train',
        transform=None,
        label_json='labels.json',
        include_unlabeled=False,
    ):
        self.transform = transform
        self.split = split.lower()
        self.data_dir = base_dir
        self.include_unlabeled = include_unlabeled

        if self.split not in ('train', 'test'):
            raise ValueError("split must be 'train' or 'test'")

        self.split_dir = os.path.join(base_dir, self.split)
        self.image_dir = os.path.join(self.split_dir, 'imgs')
        self.label_dir = os.path.join(self.split_dir, 'labels')
        self.label_json_path = self._resolve_label_json_path(label_json)
        self.labels = self._load_labels(self.label_json_path)
        self.class_names = [item['name'] for item in self.labels]
        self.class_map = {item['name']: item['classid'] for item in self.labels}
        self.color_to_class = {
            tuple(item['color']): item['classid']
            for item in self.labels
        }
        self.num_classes = max(item['classid'] for item in self.labels) + 1

        image_names = self._list_png_names(self.image_dir)
        label_names = self._list_png_names(self.label_dir)
        label_name_set = set(label_names)
        if include_unlabeled:
            sample_names = image_names
        else:
            sample_names = [name for name in image_names if name in label_name_set]

        self.sample_list = sample_names
        self.image_files = [
            os.path.join(self.image_dir, name)
            for name in self.sample_list
        ]
        self.label_files = [
            os.path.join(self.label_dir, name) if name in label_name_set else None
            for name in self.sample_list
        ]

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        image = self._load_image(image_path)
        if label_path is None:
            label = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            label = self._load_label(label_path)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = os.path.splitext(os.path.basename(image_path))[0]
        return sample

    def _resolve_label_json_path(self, label_json):
        if os.path.isabs(label_json):
            if os.path.exists(label_json):
                return label_json
            raise FileNotFoundError("Could not find {}".format(label_json))

        root_candidate = os.path.join(self.data_dir, label_json)
        if os.path.exists(root_candidate):
            return root_candidate
        else:
            raise ValueError("Could not find {} in data_dir".format(label_json))

        # train_candidate = os.path.join(
        #     self.data_dir,
        #     'train',
        #     'labels',
        #     label_json,
        # )
        # if os.path.exists(train_candidate):
        #     return train_candidate

        # candidate = os.path.join(self.label_dir, label_json)
        # if os.path.exists(candidate):
        #     return candidate

        # raise FileNotFoundError("Could not find {}".format(label_json))

    def _load_labels(self, label_json_path):
        with open(label_json_path, 'r') as f:
            labels = json.load(f)

        required_keys = {'name', 'color', 'classid'}
        for item in labels:
            if not required_keys.issubset(item):
                raise ValueError(
                    "Each label entry must contain name, color, and classid"
                )
            item['color'] = [int(value) for value in item['color']]
            item['classid'] = int(item['classid'])

        return sorted(labels, key=lambda item: item['classid'])

    def _list_png_names(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError("Directory not found: {}".format(directory))

        return sorted(
            name
            for name in os.listdir(directory)
            if name.lower().endswith('.png')
        )

    def _load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("Could not read image: {}".format(image_path))

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_label(self, label_path):
        label_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label_image is None:
            raise FileNotFoundError("Could not read label: {}".format(label_path))

        if label_image.ndim == 2:
            return label_image.astype(np.uint8)

        if label_image.shape[2] == 4:
            label_image = cv2.cvtColor(label_image, cv2.COLOR_BGRA2RGB)
        else:
            label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

        return self._color_label_to_class_ids(label_image)

    def _color_label_to_class_ids(self, label_image):
        mask = np.full(label_image.shape[:2], 255, dtype=np.uint8)

        for color, class_id in self.color_to_class.items():
            color_arr = np.array(color, dtype=np.uint8)
            matches = np.all(label_image == color_arr, axis=-1)
            mask[matches] = class_id

        unknown = mask == 255
        if np.any(unknown):
            unknown_colors, counts = np.unique(
                label_image[unknown].reshape(-1, 3),
                axis=0,
                return_counts=True,
            )

            order = np.argsort(counts)[::-1]
            preview = [
                {
                    "rgb": unknown_colors[i].tolist(),
                    "count": int(counts[i]),
                }
                for i in order[:20]
            ]

            raise ValueError(
                "Unknown RGB colors found in label mask. "
                f"Most frequent unknown colors: {preview}. "
                "Your labels.json is incomplete or mismatched."
            )

        return mask


# EndoVis2018_Dataset = EndoVis2018Dataset
