'''
densify network
tianye li
Please see LICENSE for the licensing information
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.model_aligner.base_model import BaseModel

# -----------------------------------------------------------------------------

class Model(BaseModel):

    def __init__(self,
        input_ch,
        pts_num,
        mesh_resample_path,
        mesh_resample_start=1,
        mesh_resample_end=0,
        local_architecture='v2v',
        local_voxel_dim=16,
        local_voxel_inc_list=[0.25, 0.125],
        local_net_level=3,
        global_embedding_type=None,
        enforce_no_normal=False,
        normal_scale_factor=1.0,
        **kwargs
    ):
        super(Model, self).__init__()

        # ---- properties ----
        self.input_ch = input_ch  # input feature dim
        self.pts_num = pts_num

        # precomputed upsampling matrices
        from modules.mesh_resampling import MeshResampler
        self.mr = MeshResampler(
            info_path=mesh_resample_path,
            level_start=mesh_resample_start,
            level_end=mesh_resample_end,
            normal_scale_factor=normal_scale_factor
        )

        # volumetric feature sampler
        self.fuse_ops_num = 2  # hardcoded, same as len(VolumetricFeatureSampler.fuse_ops)

        # local voxel
        self.local_architecture = local_architecture
        self.vd = self.local_voxel_dim = local_voxel_dim
        self.local_voxel_inc_list = local_voxel_inc_list
        self.local_net_level = local_net_level  # TODO: maybe we can remove this parameter

        # global mesh embedding (included in MeshResampler)
        self.global_embedding_type = global_embedding_type
        if global_embedding_type in [None, '', 'none']:
            self.ged = 0
        elif global_embedding_type in ['vt', 'uv']:
            self.ged = 2
        else:
            raise RuntimeError(f"invalid global_embedding_type = {global_embedding_type}")\

        self.enforce_no_normal = enforce_no_normal
        print(f"DEBUG: self.enforce_no_normal: {self.enforce_no_normal}")
        # import ipdb; ipdb.set_trace()

        # ---- submodules ----
        self.module_names = ['local_net']

        if self.local_architecture == 'v2v':
            from modules.v2v_alternative_depth_3 import V2VModel
            # from modules.aligner_net.v2v import V2VModel
            # self.local_net = V2VModel(
            #     input_channels=len(self.mv_fuse_ops) * (0+len(self.local_voxel_inc_list)) * self.input_ch + self.ged,
            #     output_channels=1, encdec_level=self.local_net_level)

            self.local_net = V2VModel(
                input_channels=self.fuse_ops_num * (0+len(self.local_voxel_inc_list)) * self.input_ch + self.ged,
                output_channels=1, encdec_level=self.local_net_level)

        else:
            raise RuntimeError( "unrecognizable local_architecture: %s" % ( self.local_architecture ) )

    def print_setting(self):
        print("-"*40)
        print(f"name: densify_net_upsample")
        print(f"\t- input_ch: {self.input_ch}")
        print(f"\t- pts_num: {self.pts_num}")
        print(f"\t- upsampling:")
        self.mr.print_setting()
        print(f"\t- local_net:")
        print(f"\t\t- local_architecture: {self.local_architecture}")
        print(f"\t\t- local_voxel_dim: {self.local_voxel_dim}")
        print(f"\t\t- local_voxel_inc_list: {self.local_voxel_inc_list}")
        print(f"\t\t- local_net_level: {self.local_net_level}")
        print(f"\t\t- global_embedding_type: {self.global_embedding_type}")

    # ---- differentiable processes ----

    def normalize_volume(self, volume):
        ''' normalize volume spatially (throughout volume dimensions)
        Args and Returns:
            volume: (B, L, D, D, D)
        '''
        bs, ln, dim, dim, dim = volume.shape
        volume = torch.nn.Softmax(dim=2)(
            volume.view(bs, ln, -1)).view(bs, ln, dim, dim, dim)
        return volume

    def sample_local_features(self, feats, RTs, Ks, pts_global, random_grid=False):
        '''sample volumetric features from local voxel grids
        Args:
            feats: tensor in (B, V, F, H, W), 2d features from images
            RTs: tensor in (B, V, 3, 4)
            Ks: tensor in (B, V, 3, 3)
            pts_global: tensor in (B, L, 3)
        Returns:
            local_feat_grids: (B, L, F', D, D, D), F' = F * len(self.local_voxel_inc_list) * len(self.mv_fuse_ops)
            local_grids: list of (B, L, 3, D, D, D)
            local_disps: list of (B, L, 3, D, D, D)
            local_Rot: (B, L, 3, 3)
        '''
        from modules.voxel_utils import sample_random_rotation, generate_local_grids
        bs, vn, fd, fh, fw = feats.shape
        device = feats.device
        vd = self.local_voxel_dim  # D
        ln = pts_global.shape[1]  # L, sparse init point number

        # create grid ((B,L,3,3), randomly rotated voxel grids)
        local_Rot = sample_random_rotation(bs, ln).to(device) if random_grid else None

        # sample at projected locations of local detector grids
        local_feat_grids, local_grids, local_disps = [], [], []
        for gid, grid_inc in enumerate(self.local_voxel_inc_list):

            # create grid
            this_grid, this_disp = generate_local_grids(
                vert=pts_global,
                grid_dim=vd,
                grid_inc=grid_inc,
                rotate_mat=local_Rot
            )

            # sample
            from modules.vol_feat_sampler import VolumetricFeatureSampler
            vfs = VolumetricFeatureSampler()
            this_feat_grid = vfs.forward(feats, RTs, Ks, this_grid)

            local_feat_grids.append(this_feat_grid)
            local_grids.append(this_grid)
            local_disps.append(this_disp)

        # combine
        local_feat_grids = torch.cat(local_feat_grids, dim=2)  # (B, L, F', D, D, D)

        # combine global embedding
        if self.ged > 0:
            level = self.mr.get_level(pts_global)
            ge = self.mr.vts[level][None, :, :, None, None, None].repeat(bs,1,1,vd,vd,vd).to(local_feat_grids.device)
            local_feat_grids = torch.cat((local_feat_grids, ge), dim=2)

        return local_feat_grids, local_grids, local_disps, local_Rot

    def forward(
        self,
        feats,
        RTs,
        Ks,
        pts_sparse,
        random_grid=True
    ):
        ''' perform iterative upsampling and refinement given sparse initial vertices
        Args:
            feats: (B, V, F, H, W), 2d feature maps extracted from images
            RTs: (B, V, 3, 4), camera poses
            Ks: (B, V, 3, 3), camera intrinsic matrices
            pts_sparse: (B, L, 3), the sparse initial points predicted by previous global stage
            random_grid: bool, if true, the sampling grid will be randomly rotated            
        Returns:
            pts_list: list of (B, N_i, 3), N_i increasing
        '''
        from modules.voxel_utils import compute_expectation
        bs, vn, fd, fh, fw = feats.shape
        device = feats.device
        vd = self.local_voxel_dim

        # iteratively update and refine points
        pts_lower = pts_sparse
        pts_list = []

        for idx in range(0, self.mr.level_process):

            # upsample points
            pts_higher = self.mr.upsample(pts_lower, no_normals=self.enforce_no_normal)
            pn_higher = pts_higher.shape[1]

            # sample features
            local_feat_grids, local_grids, local_disps, local_Rot = self.sample_local_features(feats, RTs, Ks, pts_higher, random_grid=random_grid)

            # predict local refinement flow
            local_cost_vol = self.local_net(local_feat_grids.view(bs*pn_higher, -1, vd, vd, vd)) # outputs (B*L, 1, 32, 32, 32)
            local_cost_vol = self.normalize_volume(local_cost_vol).view(bs, pn_higher, vd, vd, vd) # the "1"-dim is squeezed
            flow = compute_expectation( local_cost_vol, local_disps[-1] ).view(bs, pn_higher, -1) # expectation computeted on the small scale

            # update
            pts_higher = pts_higher + flow
            pts_list.append(pts_higher)
            pts_lower = pts_higher  # possibly should detach

        return pts_list