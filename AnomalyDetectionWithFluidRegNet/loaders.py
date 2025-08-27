import glob

import cv2
import nibabel as nib
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import warnings

from scipy.stats import norm
from torch.utils.data import Dataset


class FlattenedPathoDataset(Dataset):
    def __init__(self, fold, train=False, base_path='/path/to/patho/data'):
        self.fold = fold
        self.train = train

        # TODO: Change to your patients' IDs and add correct path above
        all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        test_ids = [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]

        # Code assumes the following data structure:
        # basepath / Patient_WXYZ / date1 / vol_flat.nii
        # ------------------------------- / seg_flat.nii
        #
        # date1 can for example be 20223004L, which stands for year, month, day and laterality of imaged eye
        # Laterality: L for left eyes, R for right eyes

        if train:
            ids = [item for item in all_ids if item not in test_ids[fold]]
        else:
            ids = test_ids[fold]

        self.image_dirs = {}
        self.images = {}
        self.labels = {}
        self.masks = {}

        cnt = 0
        for patient_id in ids:

            patient_path = os.path.join(base_path, 'Patient_' + str(patient_id).zfill(4))
            dates_L = []
            dates_R = []

            for date in os.listdir(patient_path):
                img_path = os.path.join(patient_path, date, 'vol_flat.nii')
                img = np.asarray(nib.load(img_path).get_fdata())
                H, W, D = img.shape
                if H == 496 and W == 512 and D == 25:
                    if 'L' in date:
                        dates_L.append(date)
                    else:
                        dates_R.append(date)

            dates_L.sort()
            dates_R.sort()

            print('Patient ', patient_id, '- Dates (left eye):', len(dates_L), ', Dates (right eye):', len(dates_R))

            for date in dates_L:
                print(date)
                img_path = os.path.join(patient_path, date, 'vol_flat.nii')
                img_vol = np.asarray(nib.load(img_path).get_fdata())

                seg_path = os.path.join(patient_path, date, 'seg_flat.nii')
                seg_vol = 1. * (np.asarray(nib.load(seg_path).get_fdata()) > 0)

                for d in range(25):
                    slice = img_vol[..., d]
                    seg_slice = seg_vol[..., d]

                    if np.min(slice) == np.max(slice):
                        print('Empty slice')
                        continue

                    slice = np.flip(slice, axis=1).copy()
                    seg_slice = np.flip(seg_slice, axis=1).copy()

                    img_vol[..., d] = GuidedFilt(slice, 1)
                    seg_vol[..., d] = seg_slice

                mask = get_fluid_mask(img_vol, seg_vol)

                self.image_dirs[cnt] = img_path
                self.images[cnt] = torch.from_numpy(img_vol[None, ...].copy()).float()
                self.labels[cnt] = torch.from_numpy(seg_vol[None, ...].copy()).float()
                self.masks[cnt] = torch.from_numpy(mask[None, ...].copy()).float()
                cnt += 1

            for date in dates_R:
                print(date)
                img_path = os.path.join(patient_path, date, 'vol_flat.nii')
                img_vol = np.asarray(nib.load(img_path).get_fdata())

                seg_path = os.path.join(patient_path, date, 'seg_flat.nii')
                seg_vol = 1. * (np.asarray(nib.load(seg_path).get_fdata()) > 0)

                for d in range(25):
                    slice = img_vol[..., d]

                    if np.min(slice) == np.max(slice):
                        print('Empty slice')
                        continue

                    img_vol[..., d] = GuidedFilt(slice, 1)

                mask = get_fluid_mask(img_vol, seg_vol)

                self.image_dirs[cnt] = img_path
                self.images[cnt] = torch.from_numpy(img_vol[None, ...].copy()).float()
                self.labels[cnt] = torch.from_numpy(seg_vol[None, ...].copy()).float()
                self.masks[cnt] = torch.from_numpy(mask[None, ...].copy()).float()
                cnt += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_dir = self.image_dirs[index]
        img = self.images[index]
        seg = self.labels[index]
        mask = self.masks[index]

        img = normalize(img)

        if self.train:
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.5, 0.5])[0] == 1:
                random_value = np.random.normal(0, 0.05)
                img = img + random_value
                img[img < 0] = 0
                img[img > 1] = 1

            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.5, 0.5])[0] == 1:
                random_matrix = np.random.normal(0, 0.05, img.shape)
                img = img + random_matrix
                img[img < 0] = 0
                img[img > 1] = 1

        return {"image": img, "label": seg, "fluid_mask": mask, "image_dir": img_dir}


class FlattenedHealthyDataset(Dataset):
    def __init__(self, train=False, base_path='/path/to/healthy/data'):

        self.train = train

        # TODO: Change to your subjects' IDs and add correct path above
        study_ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

        # Code assumes the following data structure:
        # basepath / Subject_XYZ / vol_flat_OD.nii.gz
        # ---------------------  / seg_flat_OD.nii.gz
        # ---------------------  / vol_flat_OS.nii.gz
        # ---------------------  / seg_flat_OS.nii.gz
        #
        # OD = oculus dexter
        # OS = oculus sinister


        self.image_dirs = {}
        self.images = {}
        self.labels = {}

        cnt = 0

        paths_L = []
        paths_R = []

        for item in study_ids:

            patient_id = str(item[0]).zfill(3)

            patient_path = os.path.join(base_path, patient_id)

            img_path = os.path.join(patient_path, patient_id, 'vol_flat_OD.nii.gz')
            paths_R.append(img_path)

            img_path = os.path.join(patient_path, patient_id, 'vol_flat_OS.nii.gz')
            paths_R.append(img_path)

        for img_path in paths_L:
            print(img_path)
            img_vol = np.asarray(nib.load(img_path).get_fdata())[:, :, ::2]
            seg_vol = np.asarray(nib.load(img_path.replace('vol_flat_OS.nii.gz',
                                                           'seg_flat_OS.nii.gz')).get_fdata())

            for d in range(25):
                slice = img_vol[..., d]
                seg_slice = seg_vol[..., d]

                if np.min(slice) == np.max(slice):
                    print('Empty slice')
                    continue

                slice = np.flip(slice, axis=1).copy()
                seg_slice = np.flip(seg_slice, axis=1).copy()

                img_vol[..., d] = GuidedFilt(slice, 1)
                seg_vol[..., d] = seg_slice

            self.image_dirs[cnt] = img_path
            self.images[cnt] = torch.from_numpy(img_vol[None, ...].copy()).float()
            self.labels[cnt] = torch.from_numpy(seg_vol[None, ...].copy()).float()
            cnt += 1

        for img_path in paths_R:
            print(img_path)
            img_vol = np.asarray(nib.load(img_path).get_fdata())[:, :, ::2]
            seg_vol = np.asarray(nib.load(img_path.replace('vol_flat_OD.nii.gz',
                                                           'seg_flat_OD.nii.gz')).get_fdata())

            for d in range(25):
                slice = img_vol[..., d]

                if np.min(slice) == np.max(slice):
                    print('Empty slice')
                    continue

                img_vol[..., d] = GuidedFilt(slice, 1)

            self.image_dirs[cnt] = img_path
            self.images[cnt] = torch.from_numpy(img_vol[None, ...].copy()).float()
            self.labels[cnt] = torch.from_numpy(seg_vol[None, ...].copy()).float()
            cnt += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_dir = self.image_dirs[index]
        img = self.images[index]
        seg = self.labels[index]

        img = normalize(img)

        if self.train:
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.5, 0.5])[0] == 1:
                random_value = np.random.normal(0, 0.05)
                img = img + random_value
                img[img < 0] = 0
                img[img > 1] = 1

            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.5, 0.5])[0] == 1:
                random_matrix = np.random.normal(0, 0.05, img.shape)
                img = img + random_matrix
                img[img < 0] = 0
                img[img > 1] = 1

        return {"image": img, "label": seg, "image_dir": img_dir}


# Combine
class ConcatDataset(Dataset):
    def __init__(self, patho_dataset, healthy_dataset):
        self.patho_dataset = patho_dataset
        self.patho_len = len(patho_dataset)
        self.healthy_dataset = healthy_dataset
        self.healthy_len = len(healthy_dataset)

    def __getitem__(self, i, healthy_i=None):

        # Select central B-Scans with higher probability
        norm_pdf = norm.pdf(np.arange(-1, 1+1/12, 1/12), 0, 0.5)
        norm_pdf /= norm_pdf.sum()
        slice = np.random.choice(25, p=norm_pdf)

        patho_item = self.patho_dataset[i]
        if healthy_i is None:
            rand_idx = np.random.randint(self.healthy_len)
            healthy_item = self.healthy_dataset[(i + rand_idx) % self.healthy_len]
        else:
            healthy_item = self.healthy_dataset[healthy_i]

        patho_oct = patho_item['image'][..., slice]
        patho_seg = patho_item['label'][..., slice]
        patho_file = patho_item['image_dir']
        mask = patho_item['fluid_mask'][..., slice]

        healthy_oct = healthy_item['image'][..., slice]
        healthy_seg = healthy_item['label'][..., slice]
        healthy_file = healthy_item['image_dir']

        return {"patho_image": patho_oct.float(),
                "patho_label": patho_seg.float(),
                "fluid_mask": mask.float(),
                "patho_image_dir": patho_file,
                "healthy_image": healthy_oct.float(),
                "healthy_label": healthy_seg.float(),
                "healthy_image_dir": healthy_file}

    def __len__(self):
        return self.patho_len


# Use tresholding and morphological operations to generate rough pathology segmentations
def get_fluid_mask(oct_vol, seg_vol):

    mask_vol = np.zeros_like(oct_vol)

    kernel = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]]).unsqueeze(0).unsqueeze(0)

    for d in range(25):
        oct = torch.from_numpy(oct_vol[None, None, :, :, d].copy()).float()
        retina = torch.from_numpy((seg_vol > 0)[None, None, :, :, d].copy()).float()

        q = np.quantile(oct, 0.7)
        mask = 1. * (oct > q) * retina
        tmp = 1 - ((1 - retina) + mask)
        dil = F.conv2d(tmp.double(), kernel.double(), padding=5)
        mask = torch.logical_not((dil > 0).detach())

        mask_vol[..., d] = mask[0, 0, ...].numpy()

    return mask_vol


def normalize(image, path=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            image = (image - image.min()) / (image.max() - image.min())
        except Warning as e:
            if path is not None:
                print("Error in file {}...\n".format(path), e)
            else:
                print("Error...", e)
    return image


# @ https://github.com/Utkarsh-Deshmukh/Edge-Preserving-smoothing-Filter-Comparison/blob/master/src/GuidedFilt.py
def GuidedFilt(img, r):
    eps = 0.04

    I = np.double(img)
    I = I / I.max()

    I2 = cv2.pow(I, 2)

    mean_I = cv2.boxFilter(I, -1, ((2 * r) + 1, (2 * r) + 1))
    mean_I2 = cv2.boxFilter(I2, -1, ((2 * r) + 1, (2 * r) + 1))

    cov_I = mean_I2 - cv2.pow(mean_I, 2)

    var_I = cov_I

    a = cv2.divide(cov_I, var_I + eps)
    b = mean_I - (a * mean_I)

    mean_a = cv2.boxFilter(a, -1, ((2 * r) + 1, (2 * r) + 1))
    mean_b = cv2.boxFilter(b, -1, ((2 * r) + 1, (2 * r) + 1))

    q = (mean_a * I) + mean_b;

    return q


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
