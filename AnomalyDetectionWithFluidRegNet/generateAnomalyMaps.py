import argparse
import glob
import itertools
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pickle
import torch
import warnings

warnings.filterwarnings('ignore')

from loaders import GuidedFilt, normalize
from networks import FluidRegNet_OneResolutionLevel
from ..utils import SpatialTransformer, jacobian_determinant_2d
from matplotlib import cm
from scipy.ndimage import map_coordinates


def parse_args():
    parser = argparse.ArgumentParser(description='Registration of healthy to pathological OCT images to perform '
                                                 'unsupervised segmentation of anomalies')
    parser.add_argument('gpu', help="GPU selection.")
    parser.add_argument('network_path', type=str, help="Directory to trained FluidRegNet networks.")
    parser.add_argument('result_path', type=str, help="Directory to store registration results.")
    parser.add_argument('--data_path_patho', type=str, help="Path to pathological images.")
    parser.add_argument('--data_path_healthy', type=str, help="Path to healthy images.")
    return parser.parse_args()


def main(args):

    print('Starting main.')
    card = int(args.gpu)
    torch.cuda.set_device(card)
    device = 'cuda:' + str(card)

    H, W, D, = 496, 512, 25

    transformer = SpatialTransformer((H, W), device)

    path = args.result_path
    data_path_patho = args.data_path_patho
    data_path_healthy = args.data_path_healthy

    print(path, data_path_healthy, data_path_patho)

    # IDs used for training
    test_ids = [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]

    healthy_files = glob.glob(os.path.join(data_path_healthy, '*', 'vol_flat.nii.gz'))

    thresholdJ_grow = 2
    thresholdA = -0.13

    networks = []

    colormap = cm.get_cmap('viridis', 2)
    colormap.colors[0] = np.array([colormap.colors[0][0], colormap.colors[0][1], colormap.colors[0][2], 0.])

    for fold in range(5):
        regNet = FluidRegNet_OneResolutionLevel(8).cuda()
        regNet.load_state_dict( torch.load(os.path.join(args.unet_path, "lastEpoch_split" + str(fold) + ".pt"),
                                           map_location=device)['net_state'])
        regNet.eval()

        networks.append(regNet)

    files = glob.glob(os.path.join(data_path_patho, 'Patient_*', '*', 'vol_flat.nii'))
    files.sort()

    with torch.no_grad():
        for patho_file in files:
            patient_id = int(patho_file.split(os.sep)[-4][-4:])
            print(patient_id)

            try:
                splitted = patho_file.split('/')

                patho_date = splitted[-2]
                patho_id = splitted[-4]
                print(patho_file)

                oct_img_nib = nib.load(patho_file)
                affine = oct_img_nib.affine

                patho_img_flat = np.asarray(oct_img_nib.get_fdata())
                seg_img_flat = np.asarray(patho_file.replace('vol_flat.nii', 'seg_flat.nii').get_fdata())

                # -------------------------------------------------------------------------------------------
                # -------------------------------------------------------------------------------------------

                # Check whether patient has been used for training the networks
                # If so: Use only the one network that has not been trained with the given patient (i.e. patient has
                # been in the test set for the given network)
                # Else: All five networks may be used --> is there a best one? Averaging?
                patient_id = int(patho_id[-4:])
                if patient_id in list(itertools.chain.from_iterable(test_ids)):
                    train_id = True
                    for tmp in range(5):
                        if patient_id in test_ids[tmp]:
                            fold = tmp
                            break
                else:
                    train_id = False

                anomaly_map = torch.zeros((H, W, D))
                weighted_anomaly_map = torch.zeros((H, W, D))
                sum_weight = np.zeros(D)
                fixed = torch.tensor(patho_img_flat).unsqueeze(0).cuda().float()

                for healthy_file in healthy_files:

                    healthy_img = np.asarray(nib.load(healthy_file).get_fdata())

                    for d in range(D):
                        if 'OS' in os.path.basename(healthy_file):
                            healthy_img[..., d] = np.flip(healthy_img[..., d], axis=1)
                        healthy_img[..., d] = normalize(GuidedFilt(healthy_img[..., d], 1))

                    moving = torch.tensor(healthy_img).unsqueeze(0).cuda().float()
                    diff_img = (moving - fixed).detach()

                    if train_id:
                        # Instead of looping over all B-Scans treat B-scans as samples and calculate CNN output
                        # for all slices simultaneously
                        u, app = networks[fold](moving=moving.permute(3, 0, 1, 2), fixed=fixed.permute(3, 0, 1, 2),
                                                diff=diff_img.permute(3, 0, 1, 2))
                        warped_app = transformer(app, u)
                        warped_moving = transformer(moving.permute(3, 0, 1, 2) + app, u).permute(1, 2, 3, 0)[0, ...].cpu().detach().numpy()
                        mse = ((warped_moving - fixed[0, ...].cpu().detach().numpy())**2).mean((0, 1))
                        weights = 1. / mse

                        jac_vol = np.zeros_like(moving[0, ...].cpu())
                        for slice in range(D):
                            jacobian_det = jacobian_determinant_2d(u[slice:slice+1, ...], 1, 0.3194).cpu().detach().numpy()
                            jacobian_det = np.pad(jacobian_det, 2)
                            jac_vol[..., slice] = jacobian_det

                        warped_app = warped_app.permute(1, 2, 3, 0)[0, ...].cpu().detach().numpy()
                        # Generate segmentations out of appearance maps and deformation fields
                        growingA = (warped_app < thresholdA)
                        growingJ = (jac_vol > thresholdJ_grow)
                        combined = np.logical_or(growingA, growingJ)

                        anomaly_map += 1. * combined
                        print(anomaly_map.min(), anomaly_map.max())
                        weighted_anomaly_map += weights * combined
                        sum_weight += weights

                    else:
                        for fold in range(5):
                            u, app = networks[fold](moving=moving.permute(3, 0, 1, 2), fixed=fixed.permute(3, 0, 1, 2),
                                                    diff=diff_img.permute(3, 0, 1, 2))
                            warped_app = transformer(app, u)
                            warped_moving = transformer(moving.permute(3, 0, 1, 2) + app, u).permute(1, 2, 3, 0)[0, ...].cpu().detach().numpy()
                            mse = ((warped_moving - fixed[0, ...].cpu().detach().numpy()) ** 2).mean((0, 1))
                            weights = 1. / mse

                            jac_vol = np.zeros_like(moving[0, ...].cpu())
                            for slice in range(D):
                                jacobian_det = jacobian_determinant_2d(u[slice:slice + 1, ...], 1,
                                                                       0.3194).cpu().detach().numpy()
                                jacobian_det = np.pad(jacobian_det, 2)
                                jac_vol[..., slice] = jacobian_det

                            warped_app = warped_app.permute(1, 2, 3, 0)[0, ...].cpu().detach().numpy()
                            # Generate segmentations out of appearance maps and deformation fields
                            growingA = (warped_app < thresholdA)
                            growingJ = (jac_vol > thresholdJ_grow)
                            combined = np.logical_or(growingA, growingJ)

                            anomaly_map += 1. * combined
                            weighted_anomaly_map += weights * combined
                            sum_weight += weights

                if train_id:
                    anomaly_map /= len(healthy_files)
                else:
                    anomaly_map /= (5*len(healthy_files))
                weighted_anomaly_map /= sum_weight

                anomaly_map *= seg_img_flat
                weighted_anomaly_map *= seg_img_flat

                patient_result_path = os.path.join(args.result_path, patho_id)
                if not os.path.exists(patient_result_path):
                    os.mkdir(patient_result_path)
                date_result_path = os.path.join(patient_result_path, patho_date)
                if not os.path.exists(date_result_path):
                    os.mkdir(date_result_path)

                nii_vol = nib.Nifti1Image(anomaly_map, affine)
                nii_vol.header.set_zooms((1, 1, 1))
                nib.save(nii_vol, os.path.join(date_result_path, 'anomaly_map.nii.gz'))

                nii_vol = nib.Nifti1Image(weighted_anomaly_map, affine)
                nii_vol.header.set_zooms((1, 1, 1))
                nib.save(nii_vol, os.path.join(date_result_path, 'weighted_anomaly_map.nii.gz'))

            except Exception as e:
                print(e)


if __name__ == '__main__':
    args = parse_args()
    main(args)

















