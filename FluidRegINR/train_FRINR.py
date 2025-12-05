import os.path
import cv2
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from itertools import chain

from .networks import ImpNet
from .losses import MSELoss, LNCCLoss, NCCLoss, JacobianLossCoords, GradLossCoords
from .data import make_coords_tensor, fast_trilinear_interpolation, MetricMonitor


'''
Source code for
FRINR: Pathology-Aware Implicit Neural Registration for Change Analysis in Retinal OCT Data
by J. Andresen, B. Kahrs, H. Handels and T. Kepp (BVM 2026)

This code relies in large parts on https://github.com/BrainImageAnalysis/ImpRegDec
'''


# TODO: Replace paths
data_path = '/path/to/data'
base_result_path = '/path/to/result/directory'

'''
The data is assumed to be stored in the following folder structure:
/data
--/Patient_0001
----/date1                 # format: YYYYMMDDL, i.e. year, month, date, laterality (L or R)
------vol_flat.nii         # the flattened OCT volume
----/date2
------vol_flat.nii
--/Patient_0002
...

'''

# TODO: Fill in all the patient IDs
all_ids = [1, 2, 3, 4, 5]

image_dirs_baseline = []
image_dirs_followup = []

for patient_id in all_ids:

    patient_path = os.path.join(data_path, 'Patient_' + str(patient_id).zfill(4))
    dates_L = []
    dates_R = []
    for date in os.listdir(patient_path):
        if 'L' in date:
            dates_L.append(date)
        elif 'R' in date:
            dates_R.append(date)
    dates_L.sort()
    dates_R.sort()

    usable_L = []
    for date in dates_L:
        img = np.asarray(nib.load(os.path.join(patient_path, date, 'vol_flat.nii')).get_fdata())
        H, W, D = img.shape
        if H == 496 and W == 512:# and D == 25:
            usable_L.append(date)

    usable_R = []
    for date in dates_R:
        img = np.asarray(nib.load(os.path.join(patient_path, date, 'vol_flat.nii')).get_fdata())
        H, W, D = img.shape
        if H == 496 and W == 512:# and D == 25:
            usable_R.append(date)

    print('Patient ', patient_id, '- Dates (left eye):', len(usable_L), ', Dates (right eye):', len(usable_R))

    while len(usable_L) >= 2:
        baseline_date = usable_L.pop(0)
        followup_date = usable_L[0]
        print(patient_id, baseline_date, followup_date)

        img_path_baseline = os.path.join(patient_path, baseline_date, 'vol_flat.nii')
        img_path_followup = os.path.join(patient_path, followup_date, 'vol_flat.nii')
        image_dirs_baseline.append(img_path_baseline)
        image_dirs_followup.append(img_path_followup)

    while len(usable_R) >= 2:
        baseline_date = usable_R.pop(0)
        followup_date = usable_R[0]
        print(patient_id, baseline_date, followup_date)

        img_path_baseline = os.path.join(patient_path, baseline_date, 'vol_flat.nii')
        img_path_followup = os.path.join(patient_path, followup_date, 'vol_flat.nii')
        image_dirs_baseline.append(img_path_baseline)
        image_dirs_followup.append(img_path_followup)

# ---------------------------------------------------------------------------------------------------------------------

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
card = 0
torch.cuda.set_device(card)
device = 'cuda:' + str(card)

input_dim = 3
deform_dim = 3
color_dim = 1

epochs = 1000  # one epoch corresponds to all image pixel coordinates
epochs_til_summary = 50  # generate visualizations every 50th epoch

# ------------------------- Define loss weights and initialize loss functions -----------------------------------------
alpha_ncc = 1
alpha_mse = 100
alpha_res = 200
alpha_jac = 25
alpha_grad = 0.25

ncc = NCCLoss(is_tensor=False).to(device)
lncc = LNCCLoss(win=(32, 32), n_channels=color_dim, is_tensor=False).to(device)
reg = JacobianLossCoords(add_identity=False, is_tensor=False).to(device)
mse = MSELoss().to(device)
grad = GradLossCoords().to(device)
# ---------------------------------------------------------------------------------------------------------------------

# Parameters used for cropping
top = 50
bottom = 306
crop = 8

spatial_size = (128, 248)
inv_spatial_size = (496, 256)

# Main loop to register all image pairs
for index in range(len(image_dirs_baseline)):
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    ref_path = image_dirs_followup[index]
    mov_path = image_dirs_baseline[index]

    patient_id = ref_path.split(os.sep)[-3]
    baseline_date = mov_path.split(os.sep)[-2]
    followup_date = ref_path.split(os.sep)[-2]

    nifti_img = nib.load(mov_path)
    affine = nifti_img.affine

    # Cropping to retina, if flattening height is different from the one used in the FRINR
    # paper, you will need to adapt the following two lines
    moving_orig = np.float32(np.asarray(nifti_img.get_fdata()))[top:bottom, crop:-crop]
    fixed_orig = np.float32(np.asarray(nib.load(ref_path).get_fdata()))[top:bottom, crop:-crop]

    Hm, Wm, Dm = moving_orig.shape
    Hf, Wf, Df = fixed_orig.shape
    if not Dm == Df:
        print('Shapes of moving and fixed images do not match!')
        continue
    else:
        D = Dm

    # Downsampling and slice-wise normalization
    moving = np.zeros((spatial_size[0], spatial_size[1], Dm))
    fixed = np.zeros((spatial_size[0], spatial_size[1], Df))

    for d in range(D):
        moving_orig[..., d] -= moving_orig[..., d].min()
        moving_orig[..., d] /= moving_orig[..., d].max()
        moving_orig[..., d] = moving_orig[..., d] * 2 - 1
        moving[..., d] = cv2.resize(moving_orig[..., d], (spatial_size[1], spatial_size[0])).astype(np.float32)

        fixed_orig[..., d] -= fixed_orig[..., d].min()
        fixed_orig[..., d] /= fixed_orig[..., d].max()
        fixed_orig[..., d] = fixed_orig[..., d] * 2 - 1
        fixed[..., d] = cv2.resize(fixed_orig[..., d], (spatial_size[1], spatial_size[0])).astype(np.float32)


    coords_init = make_coords_tensor(dims=(spatial_size[0], spatial_size[1], Dm))

    moving_image_vol = torch.from_numpy(moving[None, None, ...])
    fixed_image_vol = torch.from_numpy(fixed[None, None, ...])

    patient_result_path = os.path.join(base_result_path, patient_id)
    if not os.path.exists(patient_result_path):
        os.mkdir(patient_result_path)

    result_path = os.path.join(patient_result_path, baseline_date + '_' + followup_date)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    print(result_path)

    deformation_network = ImpNet(input_dim=input_dim, output_dim=deform_dim, hidden_dim=256, hidden_n_layers=5,
                                 last_layer_weights_small=True,
                                 last_layer_weights=0.0001,
                                 is_last_linear=True, input_omega_0=30.0, hidden_omega_0=30.,
                                 input_n_encoding_functions=6).to(device)

    residual_network = ImpNet(input_dim=input_dim, output_dim=color_dim, hidden_dim=256, hidden_n_layers=5,
                              last_layer_weights_small=False,
                              is_last_linear=True, input_omega_0=30.0, hidden_omega_0=30.,
                              input_n_encoding_functions=6).to(device)

    params = chain(deformation_network.parameters(), residual_network.parameters())
    optimizer = torch.optim.AdamW(lr=0.0001, params=params)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # ------------------------- Training -----------------------------------------------------------------------------

    stream = tqdm(range(epochs))
    loop_monitor = MetricMonitor()
    slices = np.arange(0, D, 1)

    for epoch in stream:

        epoch_loss_ncc = 0      # NCC and LNCC to compare moving and fixed images
        epoch_loss_mse = 0      # MSE to compare appearance adapted moving and fixed images
        epoch_loss_res = 0      # Sparsity of residual
        epoch_loss_jac = 0      # Jacobian penalty (deformation regularization)
        epoch_loss_grad = 0     # Gradient loss (deformation regularization)
        epoch_loss_all = 0

        np.random.shuffle(slices)

        # One epoch corresponds to a run through all B-scans (slow!)
        for slice in slices:
            image_coords = coords_init[np.arange(slice, spatial_size[0]*spatial_size[1]*Dm, Dm), :].to(device)
            moving_image = moving_image_vol[..., slice].to(device, dtype=torch.float32)
            fixed_image = fixed_image_vol[..., slice].to(device, dtype=torch.float32)

            flow, coords = deformation_network(image_coords)
            flow_add = coords + flow

            residual_pixels = residual_network(coords, clone=False)
            residual_image = residual_pixels.view(spatial_size[0],
                                                  spatial_size[1], color_dim).permute(2, 0, 1).unsqueeze(0)
            combined_image = residual_image + moving_image

            moved_image = fast_trilinear_interpolation(moving_image[0, 0, :, :].unsqueeze(-1),
                                                       flow_add[:, 0],
                                                       flow_add[:, 1],
                                                       flow_add[:, 2]).view(spatial_size[0],
                                                                            spatial_size[1])[None, None, :, :]
            moved_combined_image = fast_trilinear_interpolation(combined_image[0, 0, :, :].unsqueeze(-1),
                                                                flow_add[:, 0],
                                                                flow_add[:, 1],
                                                                flow_add[:, 2]).view(spatial_size[0],
                                                                                     spatial_size[1])[None, None, :, :]

            loss_all = torch.tensor(0)
            loss_ncc = (ncc(moved_image, fixed_image) + lncc(moved_image, fixed_image)) / 2
            loss_mse = mse(fixed_image, moved_combined_image)
            loss_res = torch.mean(residual_image**2)
            loss_jac = reg(coords, flow_add)
            loss_grad = grad(coords, flow)

            if alpha_ncc > 0:
                loss_all = loss_all + alpha_ncc * loss_ncc

            if alpha_mse > 0:
                loss_all = loss_all + alpha_mse * loss_mse

            if alpha_res > 0:
                loss_all = loss_all + alpha_res * loss_res

            if alpha_jac > 0:
                loss_all = loss_all + alpha_jac * loss_jac

            if alpha_grad > 0:
                loss_all = loss_all + alpha_grad * loss_grad

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            epoch_loss_ncc += alpha_ncc * loss_ncc.item()
            epoch_loss_mse += alpha_mse * loss_mse.item()
            epoch_loss_res += alpha_res * loss_res.item()
            epoch_loss_jac += alpha_jac * loss_jac.item()
            epoch_loss_grad += alpha_grad * loss_grad.item()
            epoch_loss_all += loss_all.item()

            if not epoch % epochs_til_summary and slice == D//2:
                temp_moved = moved_image.cpu().detach().squeeze().squeeze(0).numpy() * 0.5 + 0.5
                temp_recon = combined_image.cpu().detach().squeeze().squeeze(0).numpy() * 0.5 + 0.5
                temp_residual = residual_pixels.cpu().detach().view(spatial_size[0], spatial_size[1]).numpy() * 0.5 + 0.5
                temp_moved_combi = moved_combined_image.cpu().detach().squeeze().squeeze(0).numpy() * 0.5 + 0.5
                temp_moving = moving_image.cpu().detach().squeeze().squeeze(0).numpy() * 0.5 + 0.5
                temp_fixed = fixed_image.cpu().detach().squeeze().squeeze(0).numpy() * 0.5 + 0.5

                temp_moving = cv2.resize(temp_moving, inv_spatial_size).astype(np.float32)
                temp_moved = cv2.resize(temp_moved, inv_spatial_size).astype(np.float32)
                temp_recon = cv2.resize(temp_recon, inv_spatial_size).astype(np.float32)
                temp_residual = cv2.resize(temp_residual, inv_spatial_size).astype(np.float32)
                temp_moved_combi = cv2.resize(temp_moved_combi, inv_spatial_size).astype(np.float32)
                temp_fixed = cv2.resize(temp_fixed, inv_spatial_size).astype(np.float32)

                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                axes[0, 0].imshow(temp_moving, vmin=0, vmax=1, cmap='gray')
                axes[0, 1].imshow(temp_recon, vmin=0, vmax=1, cmap='gray')
                axes[0, 2].imshow(temp_residual, vmin=0, vmax=1, cmap='gray')
                axes[1, 0].imshow(temp_moved_combi, vmin=0, vmax=1, cmap='gray')
                axes[1, 1].imshow(temp_moved, vmin=0, vmax=1, cmap='gray')
                axes[1, 2].imshow(temp_fixed, vmin=0, vmax=1, cmap='gray')
                vmin = flow.min().item()
                vmax = flow.max().item()
                axes[2, 0].imshow(flow[:, 0].view(spatial_size).cpu().detach().numpy(), vmin=vmin, vmax=vmax)
                axes[2, 1].imshow(flow[:, 1].view(spatial_size).cpu().detach().numpy(), vmin=vmin, vmax=vmax)
                axes[2, 2].imshow(flow[:, 2].view(spatial_size).cpu().detach().numpy(), vmin=vmin, vmax=vmax)
                axes[2, 3].imshow(np.abs(temp_moved - temp_fixed), vmin=0, vmax=0.5, cmap='gray')

                [axi.set_xticks([]) for axi in axes.ravel()]
                [axi.set_yticks([]) for axi in axes.ravel()]

                axes[0, 0].set_title('moving')
                axes[0, 1].set_title('reconstructed')
                axes[0, 2].set_title('residual')
                axes[1, 0].set_title('moved M+R')
                axes[1, 1].set_title('moved')
                axes[1, 2].set_title('fixed')
                axes[2, 3].set_title('fixed - moved')
                axes[2, 0].set_title('df x')
                axes[2, 1].set_title('df y')
                axes[2, 2].set_title('df z')
                plt.savefig(os.path.join(result_path, 'epoch{}.png'.format(epoch + 1)), dpi=256)
                plt.close()

        scheduler.step()
        loop_monitor.update('loss', loss_all.item())
        stream.set_description(
            'Epoch: {epoch}. Train. {metric_monitor}'.format(epoch=epoch, metric_monitor=loop_monitor))

        df = pd.DataFrame(data=np.array([[loss_ncc.item(), loss_mse.item(), loss_res.item(),
                                          loss_jac.item(), loss_grad.item(), loss_all.item()]]),
                          columns=('NCC Loss', 'MSE Loss', 'Sparsity Loss',
                                   'Jacobian Loss', 'Gradient Loss', 'Loss'))
        if epoch == 0:
            df.to_csv(os.path.join(result_path, "losses.csv"))
        else:
            df.to_csv(os.path.join(result_path, "losses.csv"), mode='a', header=False)

        state_dict = {'residual_net': residual_network.state_dict(),
                      'deformation_net': deformation_network.state_dict(),
                      'optimizer_state': optimizer.state_dict(),
                      'epoch': epoch}
        torch.save(state_dict, os.path.join(result_path, 'checkpoint.pt'))
