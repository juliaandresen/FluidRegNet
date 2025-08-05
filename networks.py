import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat1, out_feat2, filtersize=3, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_feat, out_feat1, filtersize, padding=padding, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(out_feat1),
            nn.LeakyReLU(),
            nn.Conv2d(out_feat1, out_feat2, filtersize, padding=padding, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(out_feat2),
            nn.LeakyReLU()
        )

    def forward(self, input):
        output = self.layers(input)
        return output

class FluidRegNet(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.n_feat = n_feat

        self.enc0 = ConvBlock(2, n_feat, n_feat, 3)
        self.enc0_2 = ConvBlock(1, n_feat, n_feat, 3)

        self.enc1 = ConvBlock(2 * n_feat, 3 * n_feat, 3 * n_feat, 3)
        self.enc2 = ConvBlock(3 * n_feat, 4 * n_feat, 4 * n_feat, 3)
        self.enc3 = ConvBlock(4 * n_feat, 8 * n_feat, 8 * n_feat, 3)
        self.enc4 = ConvBlock(8 * n_feat, 16 * n_feat, 16 * n_feat, 3)

        self.dec1 = ConvBlock(24 * n_feat, 12 * n_feat, 8 * n_feat, 3)
        self.dec2 = ConvBlock(12 * n_feat, 8 * n_feat, 6 * n_feat, 3)
        self.dec3 = ConvBlock(9 * n_feat, 6 * n_feat, 4 * n_feat, 3)
        self.dec4 = ConvBlock(6 * n_feat, 4 * n_feat, 2 * n_feat, 3)

        self.dec_def = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 1),
            nn.LeakyReLU(),
            nn.Conv2d(n_feat, 2, 1)
        )
        self.dec_app = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 1),
            nn.LeakyReLU(),
            nn.Conv2d(n_feat, 1, 1)
        )

        self.maxPool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, moving, fixed, diff):
        output0 = self.enc0(torch.cat((moving, fixed), dim=1))
        output0_1 = self.enc0_2(diff)

        output1 = self.enc1(torch.cat((self.maxPool(output0), self.maxPool(output0_1)), dim=1))
        output2 = self.enc2(self.maxPool(output1))
        output3 = self.enc3(self.maxPool(output2))
        output4 = self.enc4(self.maxPool(output3))

        output5 = self.dec1(torch.cat([output3, self.upsample(output4)], dim=1))
        output6 = self.dec2(torch.cat([output2, self.upsample(output5)], dim=1))
        output7 = self.dec3(torch.cat([output1, self.upsample(output6)], dim=1))
        output8 = self.dec4(torch.cat([output0, output0_1, self.upsample(output7)], dim=1))

        u = self.dec_def(output8)
        a = self.dec_app(output8)

        return u, a
