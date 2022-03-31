# From: https://saic-violet.github.io/learnable-triangulation/

# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, norm='bn'):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes) if norm == 'bn' else nn.InstanceNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='bn'):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes) if norm == 'bn' else nn.InstanceNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes) if norm == 'bn' else nn.InstanceNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes) if norm == 'bn' else nn.InstanceNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, norm='bn'):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes) if norm == 'bn' else nn.InstanceNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderDecorder(nn.Module):
    def __init__(self, level=5, norm='bn'):
        super().__init__()
        self.level = level

        self.encoder_res1 = Res3DBlock(32, 64, norm=norm)
        self.encoder_res2 = Res3DBlock(64, 128, norm=norm)
        self.encoder_res3 = Res3DBlock(128, 128, norm=norm)
        self.encoder_res4 = Res3DBlock(128, 128, norm=norm)
        self.encoder_res5 = Res3DBlock(128, 128, norm=norm)

        if self.level >= 1:
            self.encoder_pool1 = Pool3DBlock(2)

        if self.level >= 2:
            self.encoder_pool2 = Pool3DBlock(2)

        if self.level >= 3:
            self.encoder_pool3 = Pool3DBlock(2)

        if self.level >= 4:
            self.encoder_pool4 = Pool3DBlock(2)

        if self.level >= 5:
            self.encoder_pool5 = Pool3DBlock(2)

        self.mid_res = Res3DBlock(128, 128, norm=norm)

        self.decoder_res5 = Res3DBlock(128, 128, norm=norm)
        self.decoder_res4 = Res3DBlock(128, 128, norm=norm)
        self.decoder_res3 = Res3DBlock(128, 128, norm=norm)
        self.decoder_res2 = Res3DBlock(128, 128, norm=norm)
        self.decoder_res1 = Res3DBlock(64, 64, norm=norm)

        if self.level >= 5:
            self.decoder_upsample5 = Upsample3DBlock(128, 128, 2, 2, norm=norm)

        if self.level >= 4:
            self.decoder_upsample4 = Upsample3DBlock(128, 128, 2, 2, norm=norm)

        if self.level >= 3:
            self.decoder_upsample3 = Upsample3DBlock(128, 128, 2, 2, norm=norm)

        if self.level >= 2:
            self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2, norm=norm)

        if self.level >= 1:
            self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2, norm=norm)

        self.skip_res1 = Res3DBlock(32, 32, norm=norm)
        self.skip_res2 = Res3DBlock(64, 64, norm=norm)
        self.skip_res3 = Res3DBlock(128, 128, norm=norm)
        self.skip_res4 = Res3DBlock(128, 128, norm=norm)
        self.skip_res5 = Res3DBlock(128, 128, norm=norm)

    def forward(self, x, verbose=False):

        def _print(data, msg):
            if verbose:
                print(f"{msg:<15}, data.shape = {data.shape}")

        _print(x, msg='start')

        skip_x1 = self.skip_res1(x)
        if self.level >= 1: x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        _print(x, msg='after encode 1')

        skip_x2 = self.skip_res2(x)
        if self.level >= 2: x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        _print(x, msg='after encode 2')

        skip_x3 = self.skip_res3(x)
        if self.level >= 3: x = self.encoder_pool3(x)
        x = self.encoder_res3(x)
        _print(x, msg='after encode 3')

        skip_x4 = self.skip_res4(x)
        if self.level >= 4: x = self.encoder_pool4(x)
        x = self.encoder_res4(x)
        _print(x, msg='after encode 4')

        skip_x5 = self.skip_res5(x)
        if self.level >= 5: x = self.encoder_pool5(x)
        x = self.encoder_res5(x)
        _print(x, msg='after encode 5')

        x = self.mid_res(x)
        _print(x, msg='after mid_res')

        x = self.decoder_res5(x)
        if self.level >= 5: x = self.decoder_upsample5(x)
        x = x + skip_x5
        _print(x, msg='after decode 5')

        x = self.decoder_res4(x)
        if self.level >= 4: x = self.decoder_upsample4(x)
        x = x + skip_x4
        _print(x, msg='after decode 4')

        x = self.decoder_res3(x)
        if self.level >= 3: x = self.decoder_upsample3(x)
        x = x + skip_x3
        _print(x, msg='after decode 3')

        x = self.decoder_res2(x)
        if self.level >= 2: x = self.decoder_upsample2(x)
        x = x + skip_x2
        _print(x, msg='after decode 2')

        x = self.decoder_res1(x)
        if self.level >= 1: x = self.decoder_upsample1(x)
        x = x + skip_x1
        _print(x, msg='after decode 1')

        return x


class V2VModel(nn.Module):
    def __init__(self, input_channels, output_channels, encdec_level=5, norm='bn'):
        super().__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7, norm=norm),
            Res3DBlock(16, 32, norm=norm),
            Res3DBlock(32, 32, norm=norm),
            Res3DBlock(32, 32, norm=norm)
        )

        self.encoder_decoder = EncoderDecorder(level=encdec_level, norm=norm)

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32, norm=norm),
            Basic3DBlock(32, 32, 1, norm=norm),
            Basic3DBlock(32, 32, 1, norm=norm),
        )

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x) # Verbose=True
        x = self.back_layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


def test():

    # data
    B, C, D = 4, 2*8, 8
    image = torch.randn(B, C, D, D, D).cuda()

    model = V2VModel(input_channels=C, output_channels=8, encdec_level=3, norm='bn').cuda()
    from utils.debug import print_network
    print_network( model )

    # forward
    out = model.forward(image) # (4,8,32,32,32)

    # print
    print(f"image.shape = {image.shape}")
    print(f"out.shape = {out.shape}")

    import ipdb; ipdb.set_trace()

if __name__ == '__main__':

    test()
