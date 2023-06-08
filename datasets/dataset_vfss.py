from torchvision import transforms
import cv2
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
import os
import random
import h5py
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


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


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        image = image.transpose(2, 0, 1)  # C H W
        label = label.transpose(2, 0, 1)  # C H W

        transform = transforms.Compose([
            transforms.Resize(self.output_size)
        ])

        transform_label = transforms.Compose([
            transforms.Resize(self.output_size)
        ])

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        image = transform(image)
        label = transform_label(label)
        image = image / 255
        label = label / 255

        sample = {'image': image, 'label': label}
        return sample


class RandomGenerator_test(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose(2, 0, 1)  # C H W
        label = label.transpose(2, 0, 1)  # C H W

        transform = transforms.Compose([
            transforms.Resize(self.output_size)
        ])

        transform_label = transforms.Compose([
            transforms.Resize(self.output_size)
        ])
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        image = transform(image)
        label = transform_label(label)
        image = image / 255
        label = label / 255

        sample = {'image': image, 'label': label}
        return sample


class VFSS_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir

        if self.split == "train":

            img_folder = os.path.join(self.data_dir + "/" + split, 'img')
            mask_folder = os.path.join(self.data_dir + "/" + split, 'mask')

        if self.split == "valid":
            img_folder = os.path.join(self.data_dir + "/" + split, 'img')
            mask_folder = os.path.join(self.data_dir + "/" + split, 'mask')

        if self.split == "test":
            img_folder = os.path.join(self.data_dir + "/" + split, 'img')
            mask_folder = os.path.join(self.data_dir + "/" + split, 'mask')

        self.img_paths = []
        self.mask_paths = []
        for p in os.listdir(img_folder):
            name = p.split('.')[0]

            self.img_paths.append(os.path.join(img_folder, name + '.jpeg'))
            self.mask_paths.append(os.path.join(mask_folder, name + '.npy'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img = self.img_paths[idx]
        mask = self.mask_paths[idx]

        image = cv2.imread(img)
        label = np.load(mask)
        label = label.transpose(1, 2, 0)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.img_paths[idx]

        return sample
