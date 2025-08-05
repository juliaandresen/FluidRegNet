import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch

from networks import FluidRegNet
from loaders import seed_worker, FlattenedRCSDatasetWithGeneratedFluidMasks
from losses import AppearanceLoss, DiceLoss, JacobianRegulariser2D, NCC, MaskedDiffusion
from utils import jacobian_determinant_2d, SpatialTransformer
from matplotlib import cm
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader



if __name__ == '__main__':

    # TODO: Enter path to store results into
    path = ""

    if not os.path.exists(os.path.join(path, "pretraining")):
        os.mkdir(os.path.join(path, "pretraining"))
    if not os.path.exists(os.path.join(path, "FluidRegNet")):
        os.mkdir(os.path.join(path, "FluidRegNet"))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    card = 0
    torch.cuda.set_device(card)
    device = 'cuda:' + str(card)

    ##############################################################
    # Set Hyperparameters                                        #
    # Initialize loss functions and spatial transformers         #
    ##############################################################

    batch_size = 10
    n_feat = 8
    n_pretrain_epochs = 200
    n_epochs = 500
    print_epoch = 10

    loss_fn_ncc = NCC()
    loss_fn_appearance = AppearanceLoss()
    loss_fn_dice = DiceLoss()

    # TODO: Adjust pixel spacing to your data
    loss_fn_diffusion = MaskedDiffusion(h1=1, h2=0.3194, device=device)
    loss_fn_jacobian = JacobianRegulariser2D(pixel_spacing=(1, 0.3194))

    # Weighting factors for loss components
    w_app = 1e-4
    w_dice = 10.
    w_diff = 1.
    w_jac = 1000.
    w_ncc = 1.

    transformer = SpatialTransformer((496, 512), device)
    transformer_label = SpatialTransformer((496, 512), device, mode='nearest')

    ##############################################################
    print('Batch Size: ' + str(batch_size))
    print('Nr. of Registration Features: ' + str(n_feat))
    print('Nr. of Epochs: ' + str(n_epochs))

    cmap = cm.get_cmap('RdYlGn_r')
    newcolors = cmap(np.linspace(0, 1, 10000))
    newcolors[0] = np.array([0., 0., 0., 1.])
    newcmp = ListedColormap(newcolors)
    del cmap

    for training_set in range(5):
        os.environ['PYTHONHASHSEED'] = str(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        dataset = FlattenedRCSDatasetWithGeneratedFluidMasks(training_set, train=True)

        g = torch.Generator()
        g.manual_seed(0)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=6,
                            pin_memory=True, shuffle=True, worker_init_fn=seed_worker, generator=g)

        ########################################################################################################
        #                                    Pre-Training                                                      #
        ########################################################################################################

        pretrain_path = os.path.join(path, 'pretraining', 'fold' + str(training_set) + '_pretraining')
        os.mkdir(pretrain_path)

        regNetPretraining = FluidRegNet(n_feat=n_feat).cuda()
        regNetPretraining.train()
        for param in regNetPretraining.dec_app:
            param.requires_grad = False

        optimizerPreTraining = torch.optim.Adam(regNetPretraining.parameters(), lr=1e-4)
        schedulerPreTraining = torch.optim.lr_scheduler.StepLR(optimizerPreTraining,
                                                               step_size=n_epochs // 10, gamma=0.8)

        for epoch in range(n_pretrain_epochs):
            print('Pretrain epoch: ' + str(epoch))
            np.random.seed(epoch)
            schedulerPreTraining.step()

            sum_loss_ncc = 0
            sum_loss_dice = 0
            sum_loss_diffusion = 0
            sum_loss_jacobian = 0
            sum_loss = 0

            #
            #
            #

            print('First half of epoch: Moving Follow-Up.')
            for batch in loader:
                followup = batch['image_followup'].cuda()
                baseline = batch['image_baseline'].cuda()
                diff = (followup - baseline).detach()

                retina_label_fu = batch['retina_followup'].cuda().float()
                retina_label_b = batch['retina_baseline'].cuda().float()

                fluid_mask = batch['fluid_mask'].cuda().float()

                # Deformable registration
                u, _ = regNetPretraining(moving=followup, fixed=baseline, diff=diff)

                warped_image = transformer(followup, u)
                warped_retina_label = transformer_label(retina_label_fu, u)

                ncc_loss = loss_fn_ncc(warped_image[:, :, :, 16:-16], baseline[:, :, :, 16:-16])
                dice_loss = loss_fn_dice(warped_retina_label[:, :, :, 16:-16], retina_label_b[:, :, :, 16:-16])
                diffusion_loss = loss_fn_diffusion(u, fluid_mask.bool())
                jacobian_loss = loss_fn_jacobian(u)
                regularizer_loss = w_dice * dice_loss + w_diff * diffusion_loss + w_jac * jacobian_loss
                loss = w_ncc * ncc_loss + regularizer_loss

                # Backpropagation
                optimizerPreTraining.zero_grad()
                loss.backward()
                optimizerPreTraining.step()

                sum_loss_ncc += ncc_loss.item()
                sum_loss_dice += dice_loss.item()
                sum_loss_diffusion += diffusion_loss.item()
                sum_loss_jacobian += jacobian_loss.item()
                sum_loss += loss.item()

            # Store figure with results every print_epoch'th epoch
            if epoch % print_epoch == 0:
                with torch.no_grad():
                    fig, ax = plt.subplots(3, 4)
                    fig.suptitle('Training, Epoch: ' + str(epoch + 1) + '\n')

                    print(batch['dir_baseline'][0], batch['dir_followup'][0])

                    ax[0, 0].imshow(baseline[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 0].title.set_text('Baseline')

                    ax[0, 1].imshow(followup[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 1].title.set_text('\nFollow-Up')

                    deformed_image1 = transformer(followup, u)
                    ax[0, 2].imshow(deformed_image1[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 2].title.set_text('Deformed FU')

                    ax[0, 3].imshow(warped_image[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 3].title.set_text('Warped FU')

                    diff_img = (followup[0, 0, ...] - baseline[0, 0, ...]).cpu().detach().numpy()
                    ax[1, 0].imshow(diff_img, cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 0].title.set_text('Diff. before')

                    ax[1, 3].imshow(
                        (warped_image[0, 0, ...] - baseline[0, 0, ...]).cpu().detach().numpy(),
                        cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 3].title.set_text('Diff. after')

                    ax[2, 0].imshow((retina_label_fu[0, 0, ...] - retina_label_b[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
                                    vmax=1, interpolation=None)
                    ax[2, 0].title.set_text('Overlap before')

                    ax[2, 1].imshow((warped_retina_label[0, 0, ...] - retina_label_b[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
                                    vmax=1, interpolation=None)
                    ax[2, 1].title.set_text('Overlap after')

                    magn = torch.sqrt(torch.square(u[0, 0, :, :]) +
                                      torch.square(u[0, 1, :, :])).cpu().detach().numpy()
                    ax[2, 2].imshow(magn, interpolation=None)
                    ax[2, 2].title.set_text('Def. Field')

                    jacobian_det = jacobian_determinant_2d(u[0:1, :, :, :], 1, 0.3194)
                    ax[2, 3].imshow(jacobian_det.cpu().detach().numpy(), interpolation=None, cmap=newcmp, vmin=-0.001, vmax=2.001)
                    ax[2, 3].title.set_text('DF Jac.')

                    [axi.set_axis_off() for axi in ax.ravel()]

                    plt.savefig(pretrain_path + "/Result_Epoch" + str(epoch + 1) + "_movingFollowup.png")
                    plt.close()

                    #
                    #

            #
            #
            #

            print('Second half of epoch: Moving Baseline.')
            for batch in loader:
                followup = batch['image_followup'].cuda()
                baseline = batch['image_baseline'].cuda()
                diff = (followup - baseline).detach()

                retina_label_fu = batch['retina_followup'].cuda().float()
                retina_label_b = batch['retina_baseline'].cuda().float()

                fluid_mask = batch['fluid_mask'].cuda().float()

                # Deformable registration
                u, _ = regNetPretraining(moving=baseline, fixed=followup, diff=diff)

                warped_image = transformer(baseline, u)
                warped_retina_label = transformer_label(retina_label_b, u)

                ncc_loss = loss_fn_ncc(warped_image[:, :, :, 16:-16], followup[:, :, :, 16:-16])
                dice_loss = loss_fn_dice(warped_retina_label[:, :, :, 16:-16], retina_label_fu[:, :, :, 16:-16])
                diffusion_loss = loss_fn_diffusion(u, fluid_mask.bool())
                jacobian_loss = loss_fn_jacobian(u)
                regularizer_loss = w_dice * dice_loss + w_diff * diffusion_loss + w_jac * jacobian_loss
                loss = w_ncc * ncc_loss + regularizer_loss

                # Backpropagation
                optimizerPreTraining.zero_grad()
                loss.backward()
                optimizerPreTraining.step()

                sum_loss_ncc += ncc_loss.item()
                sum_loss_dice += dice_loss.item()
                sum_loss += loss.item()
                sum_loss_diffusion += diffusion_loss.item()
                sum_loss_jacobian += jacobian_loss.item()

            # Store figure with results every print_epoch'th epoch
            if epoch % print_epoch == 0:
                with torch.no_grad():

                    fig, ax = plt.subplots(3, 4)
                    fig.suptitle('Training, Epoch: ' + str(epoch + 1) + '\n')

                    ax[0, 0].imshow(followup[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 0].title.set_text('Follow-Up')

                    ax[0, 1].imshow(baseline[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 1].title.set_text('\nBaseline')

                    deformed_image1 = transformer(baseline, u)
                    ax[0, 2].imshow(deformed_image1[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 2].title.set_text('Deformed B')

                    ax[0, 3].imshow(warped_image[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 3].title.set_text('Warped B')

                    diff_img = (baseline[0, 0, ...] - followup[0, 0, ...]).cpu().detach().numpy()
                    ax[1, 0].imshow(diff_img, cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 0].title.set_text('Diff. before')

                    ax[1, 3].imshow(
                        (warped_image[0, 0, ...] - followup[0, 0, ...]).cpu().detach().numpy(),
                        cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 3].title.set_text('Diff. after')

                    ax[2, 0].imshow((retina_label_b[0, 0, ...] - retina_label_fu[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
                                    vmax=1, interpolation=None)
                    ax[2, 0].title.set_text('Overlap before')

                    ax[2, 1].imshow((warped_retina_label[0, 0, ...] - retina_label_fu[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
                                    vmax=1, interpolation=None)
                    ax[2, 1].title.set_text('Overlap after')

                    magn = torch.sqrt(torch.square(u[0, 0, :, :]) +
                                      torch.square(u[0, 1, :, :])).cpu().detach().numpy()
                    ax[2, 2].imshow(magn, interpolation=None)
                    ax[2, 2].title.set_text('DF Ampl.')

                    jacobian_det = jacobian_determinant_2d(u[0:1, :, :, :], 1, 0.3194)
                    ax[2, 3].imshow(jacobian_det.cpu().detach().numpy(), interpolation=None, cmap=newcmp, vmin=-0.001, vmax=2.001)
                    ax[2, 3].title.set_text('DF Jac.')

                    [axi.set_axis_off() for axi in ax.ravel()]

                    plt.savefig(pretrain_path + "/Result_Epoch" + str(epoch + 1) + "_movingBaseline.png")
                    plt.close()

            #
            #
            #

            ncc_loss = sum_loss_ncc / (2 * len(dataset))
            dice_loss = sum_loss_dice / (2 * len(dataset))
            diffusion_loss = sum_loss_diffusion / (2 * len(dataset))
            jacobian_loss = sum_loss_jacobian / (2 * len(dataset))
            L = sum_loss / (2 * len(dataset))

            print(datetime.datetime.now())
            print('Loss: {:.4f} (NCC {:.4f}, Dice: {:.4f}, '
                  'Diffusion Reg.: {:.4f}, Jacobian Reg.: {:.4f}),'.format(L, ncc_loss, dice_loss,
                                                                           diffusion_loss, jacobian_loss, ), '\n')

            df = pd.DataFrame(data=np.array([[ncc_loss, dice_loss, diffusion_loss, jacobian_loss, L]]),
                              columns=('NCC Loss', 'Dice Loss', 'Diffusion Regularizer', 'Jacobian Regularizer',
                                       'Generator Loss'))
            if epoch == 0:
                df.to_csv(pretrain_path + "/PretrainMetrics.csv")
            else:
                df.to_csv(pretrain_path + "/PretrainMetrics.csv", mode='a', header=False)

            state = {'time': str(datetime.datetime.now()),
                     'generator_state': regNetPretraining.state_dict(),
                     'generator_name': type(regNetPretraining).__name__,
                     'optimizer_g_state': optimizerPreTraining.state_dict(),
                     'optimizer_g_name': type(optimizerPreTraining).__name__,
                     'epoch': epoch
                     }

            torch.save(state, pretrain_path + "/lastEpoch_split" + str(training_set) + ".pt")

        ########################################################################################################
        #                                    Main Training                                                     #
        ########################################################################################################

        train_path = os.path.join(path, 'FluidRegNet', 'fold' + str(training_set))
        os.mkdir(train_path)

        regNet = FluidRegNet(n_feat=n_feat).cuda()
        regNet.load_state_dict(torch.load(os.path.join(pretrain_path,
                                                       'lastEpoch_split' + str(training_set) + '.pt'))['generator_state'])
        optimizer = torch.optim.Adam(regNet.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 10, gamma=0.8)

        for epoch in range(500):
            print('Epoch: ' + str(epoch))

            np.random.seed(epoch)
            scheduler.step()

            sum_loss_ncc = 0
            sum_loss_dice = 0
            sum_loss_app = 0
            sum_loss_diffusion = 0
            sum_loss_jacobian = 0
            sum_loss = 0

            #
            #
            #

            print('First half of epoch: Moving Follow-Up.')
            for batch in loader:
                followup = batch['image_followup'].cuda()
                baseline = batch['image_baseline'].cuda()
                diff = (followup - baseline).detach()

                retina_label_fu = batch['retina_followup'].cuda().float()
                retina_label_b = batch['retina_baseline'].cuda().float()

                fluid_mask = batch['fluid_mask'].cuda().float()

                # Warm-up phase: Train only new task (appearance offset map)
                if epoch < 50:
                    for param in regNet.parameters():
                        param.requires_grad = False
                    for param in regNet.dec_app.parameters():
                        param.requires_grad = True
                elif epoch == 50:
                    for param in regNet.parameters():
                        param.requires_grad = True

                # Deformable registration
                u, app = regNet(moving=followup, fixed=baseline, diff=diff)

                warped_image = transformer(followup + app, u)
                warped_retina_label = transformer_label(retina_label_fu, u)

                ncc_loss = loss_fn_ncc(warped_image[:, :, :, 16:-16], baseline[:, :, :, 16:-16])
                app_loss = loss_fn_appearance(app)
                dice_loss = loss_fn_dice(warped_retina_label[:, :, :, 16:-16], retina_label_b[:, :, :, 16:-16])
                diffusion_loss = loss_fn_diffusion(u, fluid_mask.bool())
                jacobian_loss = loss_fn_jacobian(u)
                regularizer_loss = w_dice * dice_loss + w_diff * diffusion_loss + w_jac * jacobian_loss
                loss = w_ncc * ncc_loss + w_app * app_loss + regularizer_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_loss_ncc += ncc_loss.item()
                sum_loss_dice += dice_loss.item()
                sum_loss_app += app_loss.item()
                sum_loss_diffusion += diffusion_loss.item()
                sum_loss_jacobian += jacobian_loss.item()
                sum_loss += loss.item()

            # Store figure with results every print_epoch'th epoch
            if epoch % print_epoch == 0:
                with torch.no_grad():
                    fig, ax = plt.subplots(3, 4)
                    fig.suptitle('Training, Epoch: ' + str(epoch + 1) + '\n')

                    print(batch['dir_baseline'][0], batch['dir_followup'][0])

                    ax[0, 0].imshow(baseline[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 0].title.set_text('Baseline')

                    ax[0, 1].imshow(followup[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 1].title.set_text('\nFollow-Up')

                    deformed_image1 = transformer(followup, u)
                    ax[0, 2].imshow(deformed_image1[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 2].title.set_text('Deformed FU')

                    ax[0, 3].imshow(warped_image[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 3].title.set_text('Warped FU')

                    diff_img = (followup[0, 0, ...] - baseline[0, 0, ...]).cpu().detach().numpy()
                    ax[1, 0].imshow(diff_img, cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 0].title.set_text('Diff. before')

                    ax[1, 1].imshow((followup[0, 0, ...] + app[0, 0, ...]).cpu().detach().numpy(), cmap='gray', interpolation=None)
                    ax[1, 1].title.set_text('FU + App.')

                    ax[1, 2].imshow(app[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None)
                    ax[1, 2].title.set_text('Appearance')

                    ax[1, 3].imshow(
                        (warped_image[0, 0, ...] - baseline[0, 0, ...]).cpu().detach().numpy(),
                        cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 3].title.set_text('Diff. after')

                    ax[2, 0].imshow((retina_label_fu[0, 0, ...] - retina_label_b[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
                                    vmax=1, interpolation=None)
                    ax[2, 0].title.set_text('Overlap before')

                    ax[2, 1].imshow((warped_retina_label[0, 0, ...] - retina_label_b[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
                                    vmax=1, interpolation=None)
                    ax[2, 1].title.set_text('Overlap after')

                    magn = torch.sqrt(torch.square(u[0, 0, :, :]) +
                                      torch.square(u[0, 1, :, :])).cpu().detach().numpy()
                    ax[2, 2].imshow(magn, interpolation=None)
                    ax[2, 2].title.set_text('Def. Field')

                    jacobian_det = jacobian_determinant_2d(u[0:1, :, :, :], 1, 0.3194)
                    ax[2, 3].imshow(jacobian_det.cpu().detach().numpy(), interpolation=None, cmap=newcmp, vmin=-0.001, vmax=2.001)
                    ax[2, 3].title.set_text('DF Jac.')

                    [axi.set_axis_off() for axi in ax.ravel()]

                    plt.savefig(train_path + "/Result_Epoch" + str(epoch + 1) + "_movingFollowup.png")
                    plt.close()

            #
            #
            #

            print('Second half of epoch: Moving Baseline.')
            for batch in loader:
                followup = batch['image_followup'].cuda()
                baseline = batch['image_baseline'].cuda()
                diff = (followup - baseline).detach()

                retina_label_fu = batch['retina_followup'].cuda().float()
                retina_label_b = batch['retina_baseline'].cuda().float()

                fluid_mask = batch['fluid_mask'].cuda().float()

                # Deformable registration
                u, app = regNet(moving=baseline, fixed=followup, diff=diff)

                warped_image = transformer(baseline + app, u)
                warped_retina_label = transformer_label(retina_label_b, u)

                ncc_loss = loss_fn_ncc(warped_image[:, :, :, 16:-16], followup[:, :, :, 16:-16])
                app_loss = loss_fn_appearance(app)
                dice_loss = loss_fn_dice(warped_retina_label[:, :, :, 16:-16], retina_label_fu[:, :, :, 16:-16])
                diffusion_loss = loss_fn_diffusion(u, fluid_mask.bool())
                jacobian_loss = loss_fn_jacobian(u)
                regularizer_loss = w_dice * dice_loss + w_diff * diffusion_loss + w_jac * jacobian_loss
                loss = w_ncc * ncc_loss + w_app * app_loss + regularizer_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_loss_ncc += ncc_loss.item()
                sum_loss_dice += dice_loss.item()
                sum_loss_app += app_loss.item()
                sum_loss_diffusion += diffusion_loss.item()
                sum_loss_jacobian += jacobian_loss.item()
                sum_loss += loss.item()

            # Store figure with results every print_epoch'th epoch
            if epoch % print_epoch == 0:
                with torch.no_grad():

                    fig, ax = plt.subplots(3, 4)
                    fig.suptitle('Training, Epoch: ' + str(epoch + 1) + '\n')

                    ax[0, 0].imshow(followup[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 0].title.set_text('Follow-Up')

                    ax[0, 1].imshow(baseline[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 1].title.set_text('\nBaseline')

                    deformed_image1 = transformer(baseline, u)
                    ax[0, 2].imshow(deformed_image1[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 2].title.set_text('Deformed B')

                    ax[0, 3].imshow(warped_image[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 3].title.set_text('Warped B')

                    diff_img = (baseline[0, 0, ...] - followup[0, 0, ...]).cpu().detach().numpy()
                    ax[1, 0].imshow(diff_img, cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 0].title.set_text('Diff. before')

                    ax[1, 1].imshow((baseline[0, 0, ...] + app[0, 0, ...]).cpu().detach().numpy(), cmap='gray', interpolation=None)
                    ax[1, 1].title.set_text('B + App.')

                    ax[1, 2].imshow(app[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None)
                    ax[1, 2].title.set_text('Appearance')

                    ax[1, 3].imshow(
                        (warped_image[0, 0, ...] - followup[0, 0, ...]).cpu().detach().numpy(),
                        cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 3].title.set_text('Diff. after')

                    ax[2, 0].imshow((retina_label_b[0, 0, ...] - retina_label_fu[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
                                    vmax=1, interpolation=None)
                    ax[2, 0].title.set_text('Overlap before')

                    ax[2, 1].imshow((warped_retina_label[0, 0, ...] - retina_label_fu[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
                                    vmax=1, interpolation=None)
                    ax[2, 1].title.set_text('Overlap after')

                    magn = torch.sqrt(torch.square(u[0, 0, :, :]) +
                                      torch.square(u[0, 1, :, :])).cpu().detach().numpy()
                    ax[2, 2].imshow(magn, interpolation=None)
                    ax[2, 2].title.set_text('DF Ampl.')

                    jacobian_det = jacobian_determinant_2d(u[0:1, :, :, :], 1, 0.3194)
                    ax[2, 3].imshow(jacobian_det.cpu().detach().numpy(), interpolation=None, cmap=newcmp, vmin=-0.001, vmax=2.001)
                    ax[2, 3].title.set_text('DF Jac.')

                    [axi.set_axis_off() for axi in ax.ravel()]

                    plt.savefig(train_path + "/Result_Epoch" + str(epoch + 1) + "_movingBaseline.png")
                    plt.close()

            #
            #
            #

            ncc_loss = sum_loss_ncc / (2*len(dataset))
            dice_loss = sum_loss_dice / (2*len(dataset))
            app_loss = sum_loss_app / (2*len(dataset))
            diffusion_loss = sum_loss_diffusion / (2*len(dataset))
            jacobian_loss = sum_loss_jacobian / (2*len(dataset))
            L = sum_loss / (2 * len(dataset))

            print(datetime.datetime.now())
            print('Loss: {:.4f} (NCC {:.4f}, Dice: {:.4f}, App. Ampl.: {:.4f}, '
                  'Diffusion Reg.: {:.4f}, Jacobian Reg.: {:.4f}),'.format(L, ncc_loss, dice_loss, app_loss,
                                                                           diffusion_loss, jacobian_loss), '\n')

            df = pd.DataFrame(data=np.array([[ncc_loss, dice_loss, app_loss, diffusion_loss, jacobian_loss, L]]),
                              columns=('NCC Loss', 'Dice Loss', 'Appearance Amplitude',
                                       'Diffusion Regularizer', 'Jacobian Regularizer',
                                       'Generator Loss'))
            if epoch == 0:
                df.to_csv(train_path + "/metrics.csv")
            else:
                df.to_csv(train_path + "/metrics.csv", mode='a', header=False)

            state = {'time': str(datetime.datetime.now()),
                     'generator_state': regNet.state_dict(),
                     'generator_name': type(regNet).__name__,
                     'optimizer_g_state': optimizer.state_dict(),
                     'optimizer_g_name': type(optimizer).__name__,
                     'epoch': epoch
                     }

            torch.save(state, train_path + "/lastEpoch_split" + str(training_set) + ".pt")
