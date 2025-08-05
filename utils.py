import torch
import torch.nn as nn
import torch.nn.functional as F

def jacobian_determinant_2d(u, spacing_x, spacing_y):
    B, _, H, W = u.size()

    # dense_pix = dense_flow * (torch.Tensor([H - 1, W - 1]) / 2).view(1, 2, 1, 1).to(dense_flow.device)

    grady = nn.Conv2d(2, 2, (3, 1), padding=(1, 0), bias=False, groups=2)
    grady.weight.data[:, 0, :, 0] = torch.tensor([0.5*1/spacing_y, 0, -0.5*1/spacing_y]).view(1, 3).repeat(2, 1)
    grady.to(u.device)

    gradx = nn.Conv2d(2, 2, (1, 3), padding=(0, 1), bias=False, groups=2)
    gradx.weight.data[:, 0, 0, :] = torch.tensor([0.5*1/spacing_x, 0, -0.5*1/spacing_x]).view(1, 3).repeat(2, 1)
    gradx.to(u.device)

    with torch.no_grad():
        jacobian = torch.cat((grady(u), gradx(u)), 0) + torch.eye(2, 2).view(2, 2, 1, 1).to(u.device)
        jacobian = jacobian[:, :, 2:-2, 2:-2]
        jac_det = jacobian[0, 0, :, :] * jacobian[1, 1, :, :] - jacobian[1, 0, :, :] * jacobian[0, 1, :, :]
    return jac_det


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, device, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        #vectors = [2 * (torch.arange(0, s) / (s - 1.) - 0.5) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', grid)
        self.grid = grid.to(device)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)