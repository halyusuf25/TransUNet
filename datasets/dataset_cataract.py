import os
import random
import json
import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from glob import glob
import pandas as pd

# Existing augmentation functions (unchanged)
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

class RandomGenerator4Cataract(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # Get dimensions
        h, w = image.shape[:2]  # Image is (H, W, 3)
        if h != self.output_size[0] or w != self.output_size[1]:
            # Zoom image (3D: H, W, C) - keep channels unchanged
            image_zoom = (self.output_size[0] / h, self.output_size[1] / w, 1)
            image = zoom(image, image_zoom, order=3)
            # Zoom label (2D: H, W)
            label_zoom = (self.output_size[0] / h, self.output_size[1] / w)
            label = zoom(label, label_zoom, order=0)
        
        # Convert to tensor (RGB input for Swin-Tiny)
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (3, H, W)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


def load_split(csv_path, base_dir):
    """
    Reads a CSV file and returns lists of image paths and annotation paths.
    We assume the CSV contains only an 'img' column for the image name
    (e.g. 'case5013_01.png').
    
    Args:
        csv_path (str): Path to the CSV file (train or test).
        base_dir (str): Base directory containing 'img' and 'ann' subfolders.
    
    Returns:
        (list, list): Two lists corresponding to image files and annotation files.
    """
    df = pd.read_csv(csv_path)
    
    image_files = []
    annotation_files = []
    for full_img_path in df['imgs']:
        # Extract just the final part of the path (e.g. "case5013_01.png")
        file_name = os.path.basename(full_img_path)
        
        # Build the full paths for the image and corresponding annotation
        image_path = os.path.join(base_dir, 'img', file_name)
        annotation_path = os.path.join(base_dir, 'ann', file_name + ".json")
        
        image_files.append(image_path)
        annotation_files.append(annotation_path)

    # for img_name in df['imgs']:
    #     # Full path to the image
    #     img_path = os.path.join(base_dir, 'img', img_name)
    #     image_files.append(img_path)

    #     # Derive the annotation filename by appending '.json' to the image name
    #     # e.g. "case5013_01.png" -> "case5013_01.png.json"
    #     ann_name = img_name + ".json"
    #     ann_path = os.path.join(base_dir, 'ann', ann_name)
    #     annotation_files.append(ann_path)

    return image_files, annotation_files

# Updated dataset class for Cataract1k
class Cataract1kDataset(Dataset):
    def __init__(self, base_dir, split, transform=None, train_csv='train.csv', test_csv='test.csv'):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        
        # Define subdirectories for images and annotations
        self.image_dir = os.path.join(base_dir, "img")
        print(self.image_dir)
        self.annotation_dir = os.path.join(base_dir, "ann")
        print(self.annotation_dir)
        
        # # Load all image and annotation files
        # self.image_files = sorted(glob(os.path.join(self.image_dir, "*.png")))  # Adjust extension if needed
        # self.annotation_files = sorted(glob(os.path.join(self.annotation_dir, "*.json")))
        # Read the appropriate CSV file
        if self.split.lower() == "train":
            csv_path = os.path.join(base_dir, train_csv)
        else:
            csv_path = os.path.join(base_dir, test_csv)

        # Use the helper function to get the file paths
        self.image_files, self.annotation_files = load_split(csv_path, base_dir)

        # Ensure matching number of images and annotations
        assert len(self.image_files) == len(self.annotation_files), "Mismatch between images and annotations!"
        
        # Define class mapping (adjust based on your needs)
        self.class_map = {
            "Pupil": 1,
            "Cornea": 2,
            "Lens": 3,
            "Instruments": 4,
            # Add more classes if present in your dataset
        }
        # instruments
        self.instruments = ['Slit Knife', 'Gauge', 'Capsulorhexis Cystotome', 'Spatula', 'Phacoemulsification Tip', 'Irrigation-Aspiration', 'Lens Injector', 'Incision Knife', 'Katena Forceps', 'Capsulorhexis Forceps']
        
        # Split into train/val (e.g., 80/20)
        # total_samples = len(self.image_files)
        # train_size = int(0.8 * total_samples)
        # if self.split == "train":
        #     self.image_files = self.image_files[:train_size]
        #     self.annotation_files = self.annotation_files[:train_size]
        # else:  # val
        #     self.image_files = self.image_files[train_size:]
        #     self.annotation_files = self.annotation_files[train_size:]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Load annotation
        annotation_path = self.annotation_files[idx]
        with open(annotation_path, "r") as f:
            annotation = json.load(f)
        
        # Process annotation into a mask
        label = self.process_annotation(annotation, image.shape[:2])
        
        # Prepare sample
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        # Add case_name (e.g., filename without extension)
        sample['case_name'] = os.path.splitext(os.path.basename(image_path))[0]
        return sample
    
    def process_annotation(self, annotation, image_shape):
        """
        Convert JSON annotation to a segmentation mask based on polygon points.
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)  # Background is 0
        
        # Process each object in the annotation
        for obj in annotation.get("objects", []):
            class_title = obj.get("classTitle")
            if class_title in self.class_map:
                class_id = self.class_map[class_title]
                exterior_points = np.array(obj["points"]["exterior"], dtype=np.int32)
                
                # Fill the polygon with the class ID
                cv2.fillPoly(mask, [exterior_points], class_id)
            elif class_title in self.instruments and len(self.class_map) > 3:
                # Handle instruments similarly if needed
                class_id = 4
                exterior_points = np.array(obj["points"]["exterior"], dtype=np.int32)
                cv2.fillPoly(mask, [exterior_points], 3)

        return mask

