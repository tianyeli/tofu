"""
Volumetric Feature Sampler
tianye li
Please see LICENSE for the licensing information
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------

class VolumetricFeatureSampler(nn.Module):

    def __init__(self, padding_mode='zeros'):
        """ define a volumetric feature sampler
        """
        self.padding_mode = padding_mode

        # note: the two operations will cumumlte the feature and squared feature, independently among the feature channels
        # later in self.compute_stats(), we convert the results into means and standard deviations.
        # currently these are hardcoded operations, and all tensors in this class with some dimension "2F" are also due to this reason
        self.fuse_ops = [
            lambda feat, base: feat if base is None else feat + base,
            lambda feat, base: feat.pow(2.0) if base is None else feat.pow(2.0) + base,
        ]
        self.fuse_ops_num = len(self.fuse_ops)

    @staticmethod
    def project(points, RT, K, height, width):
        """ project the grid points into images (full perspective projection)
        Args:
        - points: (B, 3, N)
        - RT: (B, 3, 4)
        - K: (B, 3, 3)
        - height: int 
        - width: int

        Return:
        - u_coord: (B, N), value range in [-1, 1]
        - v_coord: (B, N), value range in [-1, 1]
        """
        bs, _, pn = points.shape
        ones = torch.ones(bs, 1, pn).to(points.device)
        points_aug = torch.cat((points, ones), dim=-2)  # (B, 4, N)
        KRT = torch.bmm(K, RT)  # (B, 3, 4)
        uvd = torch.bmm(KRT, points_aug)  # (B, 3, N)

        # convert the range to [-1, 1]
        u_coord = 2. * uvd[:, 0, :] / uvd[:, 2, :] / (width - 1.) - 1.
        v_coord = 2. * uvd[:, 1, :] / uvd[:, 2, :] / (height - 1.) - 1.
        return u_coord, v_coord

    @staticmethod
    def compute_stats(feat_grid, view_num, split_dim=2):
        """ convert the feature grid from 1st- and 2nd-order moments to actual average and std
        Args:
        - feat_grid: (B, G, 2F, GH, GW, GD)
        - view_num: int
        - split_dim: int, which dim to split, currently should always set to 2 (corresp. to the dim of "2F")

        Return:
        - feat_grid: (B, G, 2F, GH, GW, GD)
        """
        feat_grid /= float(view_num)
        feat_mean, feat_var = feat_grid.chunk(2, dim=split_dim)
        feat_var = feat_var - feat_mean.pow(2)
        feat_grid = torch.cat((feat_mean, feat_var), dim=split_dim)
        return feat_grid

    def forward(self, feat_maps, RTs, Ks, grids):
        """ the full forward process
        Args:
        - feat_maps: tensor in (B, V, F, H, W), batched feature map
        - RTs: tensor in (B, V, 3, 4), batched rigid transformation (camera poses)
        - Ks: tensor in (B, V, 3, 3), batched intrisic matrices
        - grids: (B, G, 3, GH, GW, GD), batched grid coordinates

        B: batch size; F: feature dim; V: view num;
        G: grid num (global stage = 1, local stage = input vertex number)
        GH, GW, GD: shape of the grid, currently all equal to grid dim D

        Return:
        - voxels: tensor in (B, G, 2F, GH, GW, GD), sampled volumetric features
        """
        bs, view_num, fd, height, width = feat_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(bs, 3, -1)               # (B, 3, G*GH*GW*GD)

        voxels_all = []
        for opid, op in enumerate(self.fuse_ops):  # operation
            voxel = None

            for vid in range(view_num):

                # projection
                u_coord, v_coord = self.project(xyzs, RTs[:, vid], Ks[:, vid], height, width)  # u_coord.shape = (1, 32768)

                # sample
                grid2d_uv = torch.stack((u_coord, v_coord), dim=2).view(bs, grid_num, -1, 2)
                feat2d_uv = F.grid_sample(feat_maps[:, vid], grid2d_uv, padding_mode=self.padding_mode)  # (B, F, G, GH*GW*GD)
                feat2d_uv = feat2d_uv.view(bs, -1, grid_num, grid_h, grid_w, grid_d)  # (B, F, G, GH, GW, GD)
                if voxel is None:
                    voxel = op(feat2d_uv, None)
                else:
                    voxel = op(feat2d_uv, voxel)

            voxels_all.append(voxel.transpose(1, 2))  # (B, G, F, GH, GW, GD)

        # convert to mean and std
        voxels = torch.cat(voxels_all, dim=2).contiguous().view(bs, grid_num, -1, grid_h, grid_w, grid_d)  # (B, G, F*len(ops), GH, GW, GD)
        voxels = self.compute_stats(voxels, view_num=view_num)  # (B, G, 2F, GH, GW, GD)
        return voxels