'''
The local stage of ToFu (base)

tianye li
Please see LICENSE for the licensing information
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_aligner.base_model import BaseModel

# -----------------------------------------------------------------------------

class Model(BaseModel):

    def __init__(self, args):
        super(Model, self).__init__()

        # ---- properties ----
        self.args = args
        self.input_ch = 3  # RGB
        self.descriptor_dim = args.descriptor_dim
        self.lmk_num = args.lmk_num

        self.mesh_resample_path = args.mesh_resample_path
        self.local_voxel_dim = args.local_voxel_dim
        self.local_voxel_inc_list = args.local_voxel_inc_list
        self.local_net_level = args.local_net_level

        # ---- submodules ----
        self.module_names = ['feature_net', 'densify_net']

        # feature extractor for 2d
        from models.model_aligner import FeatureNet2D
        self.feature_net = FeatureNet2D(
            input_ch=self.input_ch,
            output_ch=self.descriptor_dim,
            architecture=args.feature_arch
        )

        # global voxel net
        from models.model_aligner import DensifyNetUnsample
        self.densify_net = DensifyNetUnsample(
            input_ch=self.descriptor_dim,
            pts_num=self.lmk_num,
            mesh_resample_path=self.mesh_resample_path,
            mesh_resample_start=args.mesh_resample_start,
            mesh_resample_end=args.mesh_resample_end,
            local_architecture=args.local_arch,
            local_voxel_dim=self.local_voxel_dim,
            local_voxel_inc_list=self.local_voxel_inc_list,
            local_net_level=self.local_net_level,
            global_embedding_type=args.global_embedding_type,
            enforce_no_normal=args.enforce_no_normal,
            normal_scale_factor=args.normal_scale_factor
        )

    def print_setting(self):
        self.feature_net.print_setting()
        self.densify_net.print_setting()

    def forward(
        self,
        imgs,
        RTs,
        Ks,
        pts_sparse,
        random_grid=True,
    ):
        ''' The local stage of ToFu (base)
        Args:
            imgs: (B, V, 3, H', W'), input images, H', W' are orig size, as compared to feature size in H, W
            RTs: (B, V, 3, 4), camera poses
            Ks: (B, V, 3, 3), camera intrinsic matrices
            pts_sparse: (B, L, 3), the sparse initial points predicted by previous global stage
            random_grid: bool, if true, the sampling grid will be randomly rotated
        Returns:
            pts_dense: list of (B, M, 3), N_i increasing
        '''
        # meta
        bs, vn, ic, ih, iw = imgs.shape
        assert ic == self.input_ch, f"unmatched input image channel: {ic} (expected: {self.input_ch})"

        # step 1: feature extraction
        imgs = imgs.view(-1, ic, ih, iw)
        feat_2d = self.feature_net(imgs).view(bs, vn, -1, ih, iw) # (B, V, F, H, W)

        # step 2: densify network
        ret_data = self.densify_net(
            feat_2d,
            RTs,
            Ks,
            pts_sparse,
            random_grid=random_grid,
        )
        return ret_data