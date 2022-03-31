'''
voxel utilities

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

# -----------------------------------------------------------------------------

def compute_expectation(cost_vol, disp_vol):
    # cost_vol: (B, L, D, D, D), normalized to 1 with voxel dimensions
    # disp_vol: (B, L, 3, D, D, D), displacement vectors or location vectors
    bs = cost_vol.shape[0]
    # disp_vol = disp_vol[:,None,:] # (B, 1, 3, D, D, D)
    x_flow = ( cost_vol * disp_vol[:, :, 0] ).sum(-1).sum(-1).sum(-1)[:,:,None] # (B,L,1)
    y_flow = ( cost_vol * disp_vol[:, :, 1] ).sum(-1).sum(-1).sum(-1)[:,:,None] # (B,L,1)
    z_flow = ( cost_vol * disp_vol[:, :, 2] ).sum(-1).sum(-1).sum(-1)[:,:,None] # (B,L,1)
    return torch.cat((x_flow, y_flow, z_flow), dim=2) # (B,L,3)

# -----------------------------------------------------------------------------

def sample_random_rotation(batch_size, point_num):
    # returns rotation matrices in shape (B, N, 3, 3)
    bs, pn = batch_size, point_num
    from liegroups.torch import SO3
    quat = torch.randn(bs*pn, 4)
    quat = F.normalize( quat, eps=1e-6 )
    Rot = SO3.from_quaternion( quat ).mat # (B*N,3,3)
    return Rot.view(bs, pn, 3, 3)

# -----------------------------------------------------------------------------

def generate_local_grids(vert, grid_dim, grid_inc, rotate_mat=None, debug=False):
    '''generate local 3D grids around base points, according to the "center" rule
    Args:
        vert: tensor in (B, N, 3), the "translation"
        grid_dim: voxel dimension (D), typically 32
        grid_inc: scalar, in world unit (e.g. meters, cm, etc.)
        rotate_mat (optional): tensor (B, N, 3, 3)
    Returns:
        vert_grid: tensor in (B, N, 3, D, D, D)
        disp_grid: tensor in (B, N, 3, D, D, D), the displacement vectors around vert
    '''
    bs, pn, _ = vert.shape
    device = vert.device

    # single grid
    xx, yy, zz = torch.meshgrid([
        torch.arange( -grid_dim//2, grid_dim//2 ),
        torch.arange( -grid_dim//2, grid_dim//2 ),
        torch.arange( -grid_dim//2, grid_dim//2 )])

    # if grid_dim is odd, then make the grids "symmetrical" around zero
    if grid_dim % 2 == 1:
        xx, yy, zz = xx+1, yy+1, zz+1

    xx = xx.float().to(device) * float(grid_inc)
    yy = yy.float().to(device) * float(grid_inc)
    zz = zz.float().to(device) * float(grid_inc)
    disp_grid = torch.cat((xx[None,:], yy[None,:], zz[None,:]), dim=0) # (3,D,D,D)

    # apply rotation
    if rotate_mat is not None:
        rotate_mat = rotate_mat.view(bs*pn, 3, 3)
        disp_grid = disp_grid[None, :, :, :, :].repeat(bs*pn,1,1,1,1).view(bs*pn,3,-1) # (bs*pn, 3, D*D*D)
        disp_grid = torch.bmm( rotate_mat, disp_grid ) # (bs*pn, 3, D*D*D)
        disp_grid = disp_grid.view(bs, pn, 3, grid_dim, grid_dim, grid_dim)
    else:
        disp_grid = disp_grid[None,None,:,:,:,:].repeat(bs,pn,1,1,1,1)

    # translated by the base point
    vert_grid = vert[:,:,:,None,None,None] + disp_grid

    if debug:
        return vert_grid, disp_grid, xx, yy, zz
    else:
        return vert_grid, disp_grid