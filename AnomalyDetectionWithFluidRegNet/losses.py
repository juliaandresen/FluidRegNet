import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedDiffusion(nn.Module):
    def __init__(self, h1=1, h2=1, device='cuda:0'):
        super(MaskedDiffusion, self).__init__()
        self.h1 = h1
        self.h2 = h2

    def forward(self, deformation_field, mask):

        h1 = self.h1
        h2 = self.h2

        d1u1 = F.pad((deformation_field[:, 0:1, :, 2:] - deformation_field[:, 0:1, :, :-2]),
                     (1, 1, 0, 0), mode='replicate') / (2 * h1)
        d2u1 = F.pad((deformation_field[:, 1:2, :, 2:] - deformation_field[:, 1:2, :, :-2]),
                     (1, 1, 0, 0), mode='replicate') / (2 * h1)

        d1u2 = F.pad((deformation_field[:, 0:1, 2:, :] - deformation_field[:, 0:1, :-2, :]),
                     (0, 0, 1, 1), mode='replicate') / (2 * h2)
        d2u2 = F.pad((deformation_field[:, 1:2, 2:, :] - deformation_field[:, 1:2, :-2, :]),
                     (0, 0, 1, 1), mode='replicate') / (2 * h2)

        r = torch.masked_select(d1u1.square() + d2u1.square() + d1u2.square() + d2u2.square(), mask)

        return r.mean()


class Diffusion(nn.Module):
    def __init__(self, h1=1, h2=1):
        super(Diffusion, self).__init__()
        self.h1 = h1
        self.h2 = h2

    def forward(self, deformation_field):

        h1 = self.h1
        h2 = self.h2

        d1u1 = F.pad((deformation_field[:, 0:1, :, 2:] - deformation_field[:, 0:1, :, :-2]),
                     (1, 1, 0, 0), mode='replicate') / (2 * h1)
        d2u1 = F.pad((deformation_field[:, 1:2, :, 2:] - deformation_field[:, 1:2, :, :-2]),
                     (1, 1, 0, 0), mode='replicate') / (2 * h1)

        d1u2 = F.pad((deformation_field[:, 0:1, 2:, :] - deformation_field[:, 0:1, :-2, :]),
                     (0, 0, 1, 1), mode='replicate') / (2 * h2)
        d2u2 = F.pad((deformation_field[:, 1:2, 2:, :] - deformation_field[:, 1:2, :-2, :]),
                     (0, 0, 1, 1), mode='replicate') / (2 * h2)

        r = d1u1.square() + d2u1.square() + d1u2.square() + d2u2.square()

        return r.mean()


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, input, real):
        if real:
            target = torch.ones_like(input).to(input.device)
        else:
            target = torch.zeros_like(input).to(input.device)
        loss = F.binary_cross_entropy_with_logits(input, target)

        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        intersections = (inputs * targets).sum(dim=[2, 3])
        cardinalities = (inputs + targets).sum(dim=[2, 3])
        dice = ((2 * intersections.sum(dim=1) + smooth) / (cardinalities.sum(dim=1) + smooth)).mean()
        return 1 - dice


class NCC(nn.Module):
    def __init__(self):
        super(NCC, self).__init__()

    def forward(self, moving, fixed):
        eps = 1e-5
        tmp1 = torch.square(torch.sum(fixed * moving, dim=[1, 2, 3]))
        tmp2 = torch.sum(fixed ** 2, dim=[1, 2, 3])
        tmp3 = torch.sum(moving ** 2, dim=[1, 2, 3])
        ncc = (tmp1 + eps) / (tmp2 * tmp3 + eps)

        # Masked D_NCC
        d_ncc = 1 - ncc

        return d_ncc.mean()


class AppearanceLoss(nn.Module):
    def __init__(self):
        super(AppearanceLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, appearance):
        sparsity = torch.sum(torch.abs(appearance), dim=[1, 2, 3]).mean()
        amplitude = torch.sum(torch.square(self.relu(8 * appearance)), dim=[1, 2, 3]).mean()
        return 0.3 * sparsity + amplitude


class AppearanceSparsity(nn.Module):
    def __init__(self):
        super(AppearanceSparsity, self).__init__()

    def forward(self, appearance):
        sparsity = torch.sum(torch.abs(appearance), dim=[1, 2, 3]).mean()
        return sparsity


class AppearanceAmplitude(nn.Module):
    def __init__(self):
        super(AppearanceAmplitude, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, appearance):
        amplitude = torch.sum(torch.square(self.relu(appearance)), dim=[1, 2, 3]).mean()
        return amplitude


class JacobianRegulariser2D(nn.Module):
    def __init__(self, pixel_spacing=(1, 1)):
        super(JacobianRegulariser2D, self).__init__()
        self.relu = nn.ReLU()
        self.spacing = pixel_spacing

    def forward(self, displacement):
        B, _, H, W = displacement.size()
        spacing_x, spacing_y = self.spacing

        grady = nn.Conv2d(2, 2, (3, 1), padding=(1, 0), bias=False, groups=2)
        grady.weight.data[:, 0, :, 0] = torch.tensor([0.5*1/spacing_y, 0, -0.5*1/spacing_y]).view(1, 3).repeat(2, 1)
        grady.to(displacement.device)

        gradx = nn.Conv2d(2, 2, (1, 3), padding=(0, 1), bias=False, groups=2)
        gradx.weight.data[:, 0, 0, :] = torch.tensor([0.5*1/spacing_x, 0, -0.5*1/spacing_x]).view(1, 3).repeat(2, 1)
        gradx.to(displacement.device)

        jacobian = torch.cat((grady(displacement), gradx(displacement)), 0) + \
                   torch.eye(2, 2).view(2, 2, 1, 1).repeat(B, 1, 1, 1).to(displacement.device)
        jacobian = jacobian[:, :, 2:-2, 2:-2]
        jac_det = jacobian[0, 0, :, :] * jacobian[1, 1, :, :] - jacobian[1, 0, :, :] * jacobian[0, 1, :, :]

        # Incompressibility
        # r = (jac_det - 1.).pow(2)

        # Avoid folding
        r = (self.relu(-jac_det)).pow(2)

        return r.mean()