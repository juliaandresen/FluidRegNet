import cv2
import nibabel as nib
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import warnings

from scipy.ndimage import binary_dilation, zoom
from skimage.measure import label as connected_components
from torch.utils.data import Dataset

# TODO: Replace with your patients IDs
all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]             # All IDs
test_ids = [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]  # Test IDs per fold


class FlattenedRCSDatasetWithGeneratedFluidMasks(Dataset):
    def __init__(self, fold, train=False):
        self.fold = fold
        self.train = train

        # TODO: Add path to your flattened OCT data
        base_path = ""

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

        self.image_dirs_baseline = {}
        self.image_dirs_followup = {}
        self.images_baseline = {}
        self.images_followup = {}
        self.labels_baseline = {}
        self.labels_followup = {}
        self.masks = {}

        cnt = 0
        for patient_id in ids:

            patient_path = os.path.join(base_path, 'Patient_' + str(patient_id).zfill(4))
            dates_L = []
            dates_R = []
            for date in os.listdir(patient_path):
                if 'L' in date:
                    dates_L.append(date)
                elif 'R' in date:
                    dates_R.append(date)
            dates_L.sort()
            dates_R.sort()

            # TODO: Assumes fixed number of 25 B-Scans per volume
            # Can be changed to different resolution; Only restriction is that baseline and follow-up
            # images need to have the same resolution
            usable_L = []
            for date in dates_L:
                img = np.asarray(nib.load(os.path.join(patient_path, date, 'vol_flat.nii')).get_fdata())
                H, W, D = img.shape
                if H == 496 and W == 512 and D == 25:
                    usable_L.append(date)

            usable_R = []
            for date in dates_R:
                img = np.asarray(nib.load(os.path.join(patient_path, date, 'vol_flat.nii')).get_fdata())
                H, W, D = img.shape
                if H == 496 and W == 512 and D == 25:
                    usable_R.append(date)

            print('Patient ', patient_id, '- Dates (left eye):', len(usable_L), ', Dates (right eye):', len(usable_R))

            while len(usable_L) >= 2:
                baseline_date = usable_L.pop(0)
                followup_date = usable_L.pop(0)
                print(patient_id, baseline_date, followup_date)

                img_path_baseline = os.path.join(patient_path, baseline_date, 'vol_flat.nii')
                img_path_followup = os.path.join(patient_path, followup_date, 'vol_flat.nii')
                img_vol_baseline = np.asarray(nib.load(img_path_baseline).get_fdata())
                img_vol_followup = np.asarray(nib.load(img_path_followup).get_fdata())

                label_path_baseline = os.path.join(patient_path, baseline_date, 'seg_flat.nii')
                label_path_followup = os.path.join(patient_path, followup_date, 'seg_flat.nii')
                seg_vol_baseline = np.asarray(nib.load(label_path_baseline).get_fdata())
                seg_vol_followup = np.asarray(nib.load(label_path_followup).get_fdata())

                for d in range(25):

                    slice_baseline = img_vol_baseline[..., d]
                    slice_followup = img_vol_followup[..., d]

                    if np.min(slice_baseline) == np.max(slice_baseline):
                        print('Empty slice')
                        continue
                    if np.min(slice_followup) == np.max(slice_followup):
                        print('Empty slice')
                        continue

                    image_baseline = GuidedFilt(slice_baseline, 1)
                    image_followup = GuidedFilt(slice_followup, 1)

                    if image_baseline.sum() != 0 and image_followup.sum() != 0:
                        label_baseline = 1. * (seg_vol_baseline[..., d] > 0)
                        label_followup = 1. * (seg_vol_followup[..., d] > 0)

                        self.image_dirs_baseline[cnt] = img_path_baseline
                        self.image_dirs_followup[cnt] = img_path_followup
                        self.images_baseline[cnt] = image_baseline
                        self.labels_baseline[cnt] = label_baseline
                        self.images_followup[cnt] = image_followup
                        self.labels_followup[cnt] = label_followup
                        cnt += 1

            while len(usable_R) >= 2:
                baseline_date = usable_R.pop(0)
                followup_date = usable_R.pop(0)
                print(patient_id, baseline_date, followup_date)

                img_path_baseline = os.path.join(patient_path, baseline_date, 'vol_flat.nii')
                img_path_followup = os.path.join(patient_path, followup_date, 'vol_flat.nii')
                img_vol_baseline = np.asarray(nib.load(img_path_baseline).get_fdata())
                img_vol_followup = np.asarray(nib.load(img_path_followup).get_fdata())

                label_path_baseline = os.path.join(patient_path, baseline_date, 'seg_flat.nii')
                label_path_followup = os.path.join(patient_path, followup_date, 'seg_flat.nii')
                seg_vol_baseline = np.asarray(nib.load(label_path_baseline).get_fdata())
                seg_vol_followup = np.asarray(nib.load(label_path_followup).get_fdata())

                for d in range(25):

                    slice_baseline = img_vol_baseline[..., d]
                    slice_followup = img_vol_followup[..., d]

                    if np.min(slice_baseline) == np.max(slice_baseline):
                        print('Empty slice')
                        continue
                    if np.min(slice_followup) == np.max(slice_followup):
                        print('Empty slice')
                        continue

                    image_baseline = GuidedFilt(slice_baseline, 1)
                    image_followup = GuidedFilt(slice_followup, 1)

                    if image_baseline.sum() != 0 and image_followup.sum() != 0:
                        label_baseline = 1. * (seg_vol_baseline[..., d] > 0)
                        label_followup = 1. * (seg_vol_followup[..., d] > 0)

                        self.image_dirs_baseline[cnt] = img_path_baseline
                        self.image_dirs_followup[cnt] = img_path_followup
                        self.images_baseline[cnt] = image_baseline
                        self.labels_baseline[cnt] = label_baseline
                        self.images_followup[cnt] = image_followup
                        self.labels_followup[cnt] = label_followup
                        cnt += 1

        for index in range(cnt):
            if not index % 25:
                print('Generating mask for ' + self.image_dirs_baseline[index])
            oct1 = self.images_baseline[index]
            oct1 = torch.from_numpy(oct1[None, None, ...].copy()).float()
            oct2 = self.images_followup[index]
            oct2 = torch.from_numpy(oct2[None, None, ...].copy()).float()
            retina1 = self.labels_baseline[index]
            retina1 = torch.from_numpy(retina1[None, None, ...].copy()).float()
            retina2 = self.labels_followup[index]
            retina2 = torch.from_numpy(retina2[None, None, ...].copy()).float()
            self.masks[index] = get_fluid_mask(oct1, retina1, oct2, retina2)

        # Flip all images of left eyes
        for index in range(cnt):
            laterality = self.image_dirs_baseline[index][-14]
            if laterality == 'L':
                print('Flipping ' + self.image_dirs_baseline[index])
                image_baseline = self.images_baseline[index]
                image_followup = self.images_followup[index]
                label_baseline = self.labels_baseline[index]
                label_followup = self.labels_followup[index]
                mask = self.masks[index]

                image_baseline = np.flip(image_baseline, axis=1).copy()
                image_followup = np.flip(image_followup, axis=1).copy()
                label_baseline = np.flip(label_baseline, axis=1).copy()
                label_followup = np.flip(label_followup, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

                self.images_baseline[index] = image_baseline
                self.images_followup[index] = image_followup
                self.labels_baseline[index] = label_baseline
                self.labels_followup[index] = label_followup
                self.masks[index] = mask

    def __len__(self):
        return len(self.images_baseline)

    def __getitem__(self, index):
        image_baseline = self.images_baseline[index]
        image_followup = self.images_followup[index]
        label_baseline = self.labels_baseline[index]
        label_followup = self.labels_followup[index]
        dir_baseline = self.image_dirs_baseline[index]
        dir_followup = self.image_dirs_followup[index]
        mask = self.masks[index]

        image_baseline = normalize(image_baseline)
        image_followup = normalize(image_followup)

        if self.train:
            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.5, 0.5])[0] == 1:
                random_value = np.random.normal(0, 0.05)
                image_baseline = image_baseline + random_value
                image_baseline[image_baseline < 0] = 0
                image_baseline[image_baseline > 1] = 1

            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.5, 0.5])[0] == 1:
                random_value = np.random.normal(0, 0.05)
                image_followup = image_followup + random_value
                image_followup[image_followup < 0] = 0
                image_followup[image_followup > 1] = 1

            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.5, 0.5])[0] == 1:
                random_matrix = np.random.normal(0, 0.05, image_baseline.shape)
                image_baseline = image_baseline + random_matrix
                image_baseline[image_baseline < 0] = 0
                image_baseline[image_baseline > 1] = 1

            if np.random.choice([0, 1], 1, replace=True, p=[1 - 0.5, 0.5])[0] == 1:
                random_matrix = np.random.normal(0, 0.05, image_baseline.shape)
                image_followup = image_followup + random_matrix
                image_followup[image_followup < 0] = 0
                image_followup[image_followup > 1] = 1

        return {"image_baseline": torch.from_numpy(image_baseline[None, ...].copy()).float(),
                "image_followup": torch.from_numpy(image_followup[None, ...].copy()).float(),
                "retina_baseline": torch.from_numpy(label_baseline[None, ...].copy()).long(),
                "retina_followup": torch.from_numpy(label_followup[None, ...].copy()).long(),
                "fluid_mask": torch.from_numpy(mask[None, ...].copy()).long(),
                "dir_baseline": dir_baseline, "dir_followup": dir_followup}


def get_fluid_mask(oct1, retina1, oct2, retina2):
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

    q1 = np.quantile(oct1, 0.7)
    q2 = np.quantile(oct2, 0.7)

    mask1 = 1. * (oct1 > q1) * retina1
    mask2 = 1. * (oct2 > q2) * retina2

    tmp1 = 1 - ((1 - retina1) + mask1)
    tmp2 = 1 - ((1 - retina2) + mask2)

    dil1 = F.conv2d(tmp1.double(), kernel.double(), padding=5)
    dil2 = F.conv2d(tmp2.double(), kernel.double(), padding=5)

    mask = torch.logical_not(torch.logical_or(dil1 > 0, dil2 > 0).detach())

    return mask[0, 0, ...].detach().numpy()


def normalize(image, path=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            image = (image - image.min()) / (image.max() - image.min())
            #image = image - image.mean()
            #image = image / image.std()
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


def find_new_fluids(label_baseline, label_followup, lesionThresholdSize = 16):
    # Remove fluids that are smaller than lesionThresholdSize
    if np.any(label_baseline > 1):
        components_b = connected_components(1. * (label_baseline > 1))
        fluid_ids = np.unique(components_b)
        removed = 0
        for fluid_id in fluid_ids[1:]:
            size = (components_b == fluid_id).sum()
            if size < lesionThresholdSize:
                label_baseline[components_b == fluid_id] = 1
                components_b[components_b == fluid_id] = 0
                removed += 1
        print('Removed ' + str(removed) + ' out of ' + str(len(fluid_ids) - 1) + ' pathologies in baseline.')

    if np.any(label_followup > 1):
        components_f = connected_components(1. * (label_followup > 1))
        fluid_ids = np.unique(components_f)
        removed = 0
        for fluid_id in fluid_ids[1:]:
            size = (components_f == fluid_id).sum()
            if size < lesionThresholdSize:
                label_followup[components_f == fluid_id] = 1
                components_f[components_f == fluid_id] = 0
                removed += 1
        print('Removed ' + str(removed) + ' out of ' + str(len(fluid_ids) - 1) + ' pathologies in followup.')

    labels_b = np.unique(label_baseline)
    labels_f = np.unique(label_followup)
    labels_only_in_FU = list(set(list(labels_f)).difference(list(labels_b)))

    binary_label_baseline = label_baseline > 1
    binary_label_followup = label_followup > 1

    structure_element = np.array([[0, 0, 0, 1, 0, 0, 0],
                                  [0, 1, 1, 1, 1, 1, 0],
                                  [0, 1, 1, 1, 1, 1, 0],
                                  [1, 1, 1, 1, 1, 1, 1],
                                  [0, 1, 1, 1, 1, 1, 0],
                                  [0, 1, 1, 1, 1, 1, 0],
                                  [0, 0, 0, 1, 0, 0, 0]])
    dilated_baseline = binary_dilation(binary_label_baseline, structure=structure_element)
    dilated_followup = binary_dilation(binary_label_followup, structure=structure_element)

    # No fluid in baseline, fluid in FU
    if not binary_label_baseline.sum() and binary_label_followup.sum():
        # All fluids present in FU are new
        new_fluid_segm = binary_label_followup * label_followup

    # Fluids in both time points
    elif binary_label_baseline.sum() and binary_label_followup.sum():

        # No overlap
        if not np.logical_and(dilated_baseline, dilated_followup).sum():
            #print('Fluids in both time points, no overlap.')
            #print('Adding slice ' + str(d) + ' to new fluids dataset.')

            # Assuming that all fluids in FU are new since there is no overlap
            # with fluid in baseline
            new_fluid_segm = binary_label_followup * label_followup

        # Overlap between fluids of baseline and follow-up
        else:
            new_fluid_segm_binary = np.zeros_like(label_followup).astype('bool')

            components_f = connected_components(1. * dilated_followup)
            fluid_ids_f = np.unique(components_f)
            for fluid_id in fluid_ids_f[1:]:
                dilated_fluid = (components_f == fluid_id)

                # No overlap between individual fluid in dilated FU and dilated fluids in baseline
                if not np.logical_and(dilated_baseline, dilated_fluid).sum():
                    new_fluid_segm_binary = np.logical_or(new_fluid_segm_binary,
                                                          np.logical_and(binary_label_followup, dilated_fluid))

            # Lesion type present in FU that is not present in baseline
            # --> New fluid
            if len(labels_only_in_FU):
                for label in labels_only_in_FU:
                    new_fluid_segm_binary = np.logical_or(new_fluid_segm_binary, label_followup == label)

            new_fluid_segm = new_fluid_segm_binary * label_followup

    # No fluid in either time point, or only disappearing lesions (no new ones)
    else:
        new_fluid_segm = np.zeros_like(label_followup)

    return new_fluid_segm
