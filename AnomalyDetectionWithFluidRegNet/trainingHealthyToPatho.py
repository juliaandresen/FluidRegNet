import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch

from loaders import ConcatDataset, FlattenedHealthyDataset, FlattenedPathoDataset, seed_worker
from losses import AppearanceAmplitude, AppearanceSparsity, DiceLoss, JacobianRegulariser2D, NCC, MaskedDiffusion
from networks import FluidRegNet_OneResolutionLevel
from ..utils import jacobian_determinant_2d, SpatialTransformer
from matplotlib import cm
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # TODO
    path = "/add/your/result/path"

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    card = 13
    torch.cuda.set_device(card)
    device = 'cuda:' + str(card)

    ##############################################################
    #                     Set Parameters                         #
    ##############################################################


    batch_size = 10
    n_feat = 8
    n_pretrain_epochs = 200 * 5
    n_epochs = 500 * 5
    print_epoch = 20

    loss_fn_ncc = NCC()

    loss_fn_appearance_ampl = AppearanceAmplitude()
    w_app_ampl = 5e-3

    loss_fn_appearance_sparse = AppearanceSparsity()
    w_app_sparse = 1.6e-5

    loss_fn_dice = DiceLoss()
    w_dice = 3000.

    loss_fn_diffusion = MaskedDiffusion(h1=1, h2=0.3194, device=device)
    w_diff = .3

    loss_fn_jacobian = JacobianRegulariser2D(pixel_spacing=(1, 0.3194))
    w_jac = 1000

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

        print('Generating pathology dataset:')
        patho_dataset = FlattenedPathoDataset(training_set, train=True)
        print('Generating healthy dataset:')
        healthy_dataset = FlattenedHealthyDataset(train=True)

        dataset = ConcatDataset(patho_dataset, healthy_dataset)

        g = torch.Generator()
        g.manual_seed(0)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=6,
                            pin_memory=True, shuffle=True, worker_init_fn=seed_worker, generator=g)

        ########################################################################################################
        #                                    Pre-Training                                                      #
        ########################################################################################################

        pretrain_path = os.path.join(path, 'pretraining', 'fold' + str(training_set))
        if not os.path.exists(pretrain_path):
            os.mkdir(pretrain_path)

            regNetPretraining = FluidRegNet_OneResolutionLevel(n_feat=n_feat).cuda()
            regNetPretraining.train()
            for param in regNetPretraining.dec_app.parameters():
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

                for batch in loader:
                    fixed = batch['patho_image'].cuda()
                    moving = batch['healthy_image'].cuda()
                    diff = (moving - fixed).detach()

                    retina_fixed = batch['patho_label'].cuda().float()
                    retina_moving = batch['healthy_label'].cuda().float()

                    fluid_mask = batch['fluid_mask'].cuda().float()

                    # Deformable registration
                    u, _ = regNetPretraining(moving=moving, fixed=fixed, diff=diff)

                    warped_image = transformer(moving, u)
                    warped_retina = transformer_label(retina_moving, u)

                    ncc_loss = loss_fn_ncc(warped_image[:, :, :, 16:-16], fixed[:, :, :, 16:-16])
                    dice_loss = loss_fn_dice(warped_retina[:, :, :, 16:-16], retina_fixed[:, :, :, 16:-16])
                    diffusion_loss = loss_fn_diffusion(u, fluid_mask.bool())
                    jacobian_loss = loss_fn_jacobian(u)

                    regularizer_loss = w_dice * dice_loss + w_diff * diffusion_loss + w_jac * jacobian_loss

                    loss = ncc_loss + regularizer_loss

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

                        print(batch['patho_image_dir'][0], batch['healthy_image_dir'][0])

                        ax[0, 0].imshow(fixed[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                        vmin=0, vmax=1)
                        ax[0, 0].title.set_text('Fixed')

                        ax[0, 1].imshow(moving[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                        vmin=0, vmax=1)
                        ax[0, 1].title.set_text('\nMoving')

                        deformed_image1 = transformer(moving, u)
                        ax[0, 2].imshow(deformed_image1[0, 0, ...].cpu().detach().numpy(),
                                        cmap='gray', interpolation=None, vmin=0, vmax=1)
                        ax[0, 2].title.set_text('Deformed')

                        ax[0, 3].imshow(warped_image[0, 0, ...].cpu().detach().numpy(),
                                        cmap='gray', interpolation=None, vmin=0, vmax=1)
                        ax[0, 3].title.set_text('Adapted')

                        diff_img = (moving[0, 0, ...] - fixed[0, 0, ...]).cpu().detach().numpy()
                        ax[1, 0].imshow(diff_img, cmap='gray', interpolation=None,
                                        vmin=diff_img.min(), vmax=diff_img.max())
                        ax[1, 0].title.set_text('Diff. before')

                        ax[1, 3].imshow(
                            (warped_image[0, 0, ...] - fixed[0, 0, ...]).cpu().detach().numpy(),
                            cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                        ax[1, 3].title.set_text('Diff. after')

                        ax[2, 0].imshow((retina_moving[0, 0, ...] - retina_fixed[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
                                        vmax=1, interpolation=None)
                        ax[2, 0].title.set_text('Overlap before')

                        ax[2, 1].imshow((warped_retina[0, 0, ...] - retina_fixed[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
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

                        plt.savefig(pretrain_path + "/Result_Epoch" + str(epoch + 1) + ".png")
                        plt.close()

                        #
                        #

                ncc_loss = sum_loss_ncc / len(dataset)
                dice_loss = sum_loss_dice / len(dataset)
                diffusion_loss = sum_loss_diffusion / len(dataset)
                jacobian_loss = sum_loss_jacobian / len(dataset)
                full_loss = sum_loss / len(dataset)

                print(datetime.datetime.now())
                print('Generator Loss: {:.4f} (NCC {:.4f}, Dice: {:.4f}, '
                      'Diffusion Reg.: {:.4f}, Jacobian Reg.: {:.4f}),'.format(full_loss, ncc_loss, dice_loss,
                                                                               diffusion_loss, jacobian_loss,), '\n')

                df = pd.DataFrame(data=np.array([[ncc_loss, dice_loss, diffusion_loss, jacobian_loss, full_loss]]),
                                  columns=('NCC Loss', 'Dice Loss', 'Diffusion Regularizer', 'Jacobian Regularizer', 'Loss'))
                if epoch == 0:
                    df.to_csv(pretrain_path + "/PretrainMetrics.csv")
                else:
                    df.to_csv(pretrain_path + "/PretrainMetrics.csv", mode='a', header=False)

                state = {'time': str(datetime.datetime.now()),
                         'net_state': regNetPretraining.state_dict(),
                         'net_name': type(regNetPretraining).__name__,
                         'optimizer_state': optimizerPreTraining.state_dict(),
                         'optimizer_name': type(optimizerPreTraining).__name__,
                         'epoch': epoch
                         }

                torch.save(state, pretrain_path + "/lastEpoch_split" + str(training_set) + ".pt")

        ########################################################################################################
        #                                    Main Training                                                     #
        ########################################################################################################

        os.environ['PYTHONHASHSEED'] = str(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        train_path = os.path.join(path, 'fold' + str(training_set))
        os.mkdir(train_path)

        regNet = FluidRegNet_OneResolutionLevel(n_feat=n_feat).cuda()
        regNet.load_state_dict(torch.load(os.path.join(pretrain_path,
                                                       'lastEpoch_split' + str(training_set) + '.pt'))['net_state'])
        optimizer = torch.optim.Adam(regNet.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 10, gamma=0.8)

        for epoch in range(n_epochs):
            print('Train epoch: ' + str(epoch))
            np.random.seed(epoch)
            scheduler.step()

            sum_loss_ncc = 0
            sum_loss_dice = 0
            sum_loss_app = 0
            sum_loss_diffusion = 0
            sum_loss_jacobian = 0
            sum_loss = 0

            ## Warm-up phase: Train only appearance branch
            # if epoch < n_epochs // 10:
            #    for param in regNet.parameters():
            #        param.requires_grad = False
            #    for param in regNet.dec_app.parameters():
            #        param.requires_grad = True
            # elif epoch == n_epochs // 10:
            #    for param in regNet.parameters():
            #        param.requires_grad = True

            for batch in loader:
                fixed = batch['patho_image'].cuda()
                moving = batch['healthy_image'].cuda()
                diff = (moving - fixed).detach()

                retina_fixed = batch['patho_label'].cuda().float()
                retina_moving = batch['healthy_label'].cuda().float()

                fluid_mask = batch['fluid_mask'].cuda().float()

                # Deformable registration
                u, app = regNet(moving=moving, fixed=fixed, diff=diff)

                warped_image = transformer(moving + app, u)
                warped_retina = transformer_label(retina_moving, u)

                ncc_loss = loss_fn_ncc(warped_image[:, :, :, 16:-16], fixed[:, :, :, 16:-16])
                dice_loss = loss_fn_dice(warped_retina[:, :, :, 16:-16], retina_fixed[:, :, :, 16:-16])
                app_loss = w_app_ampl * loss_fn_appearance_ampl(app) + w_app_sparse * loss_fn_appearance_sparse(app)
                diffusion_loss = loss_fn_diffusion(u, fluid_mask.bool())
                jacobian_loss = loss_fn_jacobian(u)

                regularizer_loss = w_dice * dice_loss + w_diff * diffusion_loss + w_jac * jacobian_loss

                loss = ncc_loss + app_loss + regularizer_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_loss_ncc += ncc_loss.item()
                sum_loss_app += app_loss.item()
                sum_loss_dice += dice_loss.item()
                sum_loss += loss.item()
                sum_loss_diffusion += diffusion_loss.item()
                sum_loss_jacobian += jacobian_loss.item()

            # Store figure with results every print_epoch'th epoch
            if epoch % print_epoch == 0:
                with torch.no_grad():
                    fig, ax = plt.subplots(3, 4)
                    fig.suptitle('Training, Epoch: ' + str(epoch + 1) + '\n')

                    print(batch['patho_image_dir'][0], batch['healthy_image_dir'][0])

                    ax[0, 0].imshow(fixed[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 0].title.set_text('Fixed')

                    ax[0, 1].imshow(moving[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 1].title.set_text('\nMoving')

                    deformed_image1 = transformer(moving, u)
                    ax[0, 2].imshow(deformed_image1[0, 0, ...].cpu().detach().numpy(),
                                    cmap='gray', interpolation=None, vmin=0, vmax=1)
                    ax[0, 2].title.set_text('Deformed')

                    ax[0, 3].imshow(warped_image[0, 0, ...].cpu().detach().numpy(),
                                    cmap='gray', interpolation=None, vmin=0, vmax=1)
                    ax[0, 3].title.set_text('Adapted')

                    diff_img = (moving[0, 0, ...] - fixed[0, 0, ...]).cpu().detach().numpy()
                    ax[1, 0].imshow(diff_img, cmap='gray', interpolation=None,
                                    vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 0].title.set_text('Diff. before')

                    ax[1, 1].imshow((moving[0, 0, ...] + app[0, 0, ...]).cpu().detach().numpy(), cmap='gray',
                                    interpolation=None)
                    ax[1, 1].title.set_text('M + A')

                    ax[1, 2].imshow(app[0, 0, ...].cpu().detach().numpy(), cmap='gray', interpolation=None)
                    ax[1, 2].title.set_text('Appearance')

                    ax[1, 3].imshow(
                        (warped_image[0, 0, ...] - fixed[0, 0, ...]).cpu().detach().numpy(),
                        cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 3].title.set_text('Diff. after')

                    ax[2, 0].imshow((retina_moving[0, 0, ...] - retina_fixed[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
                                    vmax=1, interpolation=None)
                    ax[2, 0].title.set_text('Overlap before')

                    ax[2, 1].imshow((warped_retina[0, 0, ...] - retina_fixed[0, 0, ...]).cpu().detach().numpy(), vmin=-1,
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

                    plt.savefig(train_path + "/Result_Epoch" + str(epoch + 1) + ".png")
                    plt.close()

                    #
                    #

            ncc_loss = sum_loss_ncc / len(dataset)
            app_loss = sum_loss_app / len(dataset)
            dice_loss = sum_loss_dice / len(dataset)
            diffusion_loss = sum_loss_diffusion / len(dataset)
            jacobian_loss = sum_loss_jacobian / len(dataset)
            full_loss = sum_loss / len(dataset)

            print(datetime.datetime.now())
            print('Generator Loss: {:.4f} (NCC {:.4f}, Dice: {:.4f}, App. Loss: {:.4f}, '
                  'Diffusion Reg.: {:.4f}, Jacobian Reg.: {:.4f}),'.format(full_loss, ncc_loss, dice_loss, app_loss,
                                                                           diffusion_loss, jacobian_loss,), '\n')

            df = pd.DataFrame(data=np.array([[ncc_loss, dice_loss, app_loss, diffusion_loss, jacobian_loss, full_loss]]),
                              columns=('NCC Loss', 'Dice Loss', 'Appearance Loss', 'Diffusion Regularizer', 'Jacobian Regularizer', 'Loss'))
            if epoch == 0:
                df.to_csv(train_path + "/TrainMetrics.csv")
            else:
                df.to_csv(train_path + "/TrainMetrics.csv", mode='a', header=False)

            state = {'time': str(datetime.datetime.now()),
                     'net_state': regNet.state_dict(),
                     'net_name': type(regNet).__name__,
                     'optimizer_state': optimizer.state_dict(),
                     'optimizer_name': type(optimizer).__name__,
                     'epoch': epoch
                     }

            torch.save(state, train_path + "/lastEpoch_split" + str(training_set) + ".pt")
