'''
The global stage of ToFu (base)
tianye li
Please see LICENSE for the licensing information
'''

import os
from os.path import join
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.model_aligner.base_model import BaseModel

# -----------------------------------------------------------------------------

class Model(BaseModel):

    def __init__(self, args):
        super(Model, self).__init__()

        # ---- properties ----
        self.args = args
        self.input_ch = 3
        self.descriptor_dim = args.descriptor_dim
        self.lmk_num = args.lmk_num

        self.global_voxel_dim = args.global_voxel_dim
        self.global_voxel_inc = args.global_voxel_inc
        self.global_origin = args.global_origin
        self.norm = args.norm

        # ---- submodules ----
        self.module_names = ['feature_net', 'sparse_point_net']

        # feature extractor for 2d
        from models.model_aligner import FeatureNet2D
        self.feature_net = FeatureNet2D(
            input_ch=self.input_ch, output_ch=self.descriptor_dim, architecture=args.feature_arch)

        # global voxel net
        from models.model_aligner import SparsePointNetGlobalOnly
        self.sparse_point_net = SparsePointNetGlobalOnly(
            input_ch=self.descriptor_dim, pts_num=self.lmk_num, global_architecture=args.global_arch,
            global_voxel_dim=self.global_voxel_dim, global_voxel_inc=self.global_voxel_inc, global_origin=self.global_origin, norm=self.norm)

    def print_setting(self):
        self.feature_net.print_setting()
        self.sparse_point_net.print_setting()

    def forward(self, imgs, RTs, Ks, random_grid=True):
        '''compute 2d feature maps given multiview images, and create plane sweep feature volume
        Args:
            imgs: (B, V, 3, H', W'). H', W' are orig size, as compared to feature size in H, W
            RTs: (B, V, 3, 4)
            Ks: (B, V, 3, 3)
        Returns:
            pts_global: (B, L, 3)
            pts_refined: (B, L, 3)
            global_cost_vol: (B, L, Dg, Dg, Dg)
            global_grid: (B, 1, 3, Dg, Dg, Dg)
            global_disp: (B, 1, 3, Dg, Dg, Dg)
            global_Rot: (B, 1, 3, 3)
            local_cost_vol: (B, L, D, D, D)
            local_grids: list of (B, L, 3, D, D, D)
            local_disps: list of (B, L, 3, D, D, D)
            local_Rot: (B, L, 3, 3)
        '''
        # meta
        bs, vn, ic, ih, iw = imgs.shape
        device = imgs.device
        assert ic == self.input_ch, f"unmatched input image channel: {ic} (expected: {self.input_ch})"

        # step 1: feature extraction
        imgs = imgs.view(-1, ic, ih, iw)
        feat_2d = self.feature_net(imgs)
        feat_2d = feat_2d.view([bs, vn] + list(feat_2d.shape[-3:]))  # .view(bs, vn, -1, ih, iw) # (B, V, F, H', W')

        # if True:
        #     # do this before the intrinsic matrix rescaling to make it work for the original images
        #     debug_root = '/home/tli/Dropbox/DenseFaceTracking/debug/open_tofu_code/vol_feat_sample/debug_data'
        #     np.save(join(debug_root, "imgs.npy"), imgs.detach().cpu().numpy())
        #     np.save(join(debug_root, "feat_2d.npy"), feat_2d.detach().cpu().numpy())
        #     np.save(join(debug_root, "RTs.npy"), RTs.detach().cpu().numpy())
        #     np.save(join(debug_root, "Ks.npy"), Ks.detach().cpu().numpy())
        #     print(f"imgs, Ks, RTs saved")
        #     import ipdb; ipdb.set_trace()

        # (optional): when feature map is downsized, adjust the intrisics matrics
        _, _, fc, fh, fw = feat_2d.shape
        if fh != ih:
            downsize = float(fh) / float(ih)  # e.g. 0.25
            Ks[:, :, 0:2, :] = downsize * Ks[:, :, 0:2, :]

        # step 2: sparse point prediction
        pts_global, \
            global_cost_vol, global_grid, global_disp, global_Rot = self.sparse_point_net(feat_2d, RTs, Ks, random_grid=random_grid)

        return pts_global, \
            global_cost_vol, global_grid, global_disp, global_Rot