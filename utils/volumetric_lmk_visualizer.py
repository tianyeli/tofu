'''
global landmark visualizer

tianye li
Please see LICENSE for the licensing information
'''
import argparse
import torch
import torch.nn.parallel
import torch.nn as nn
import numpy as np
import imageio
import time
import os
from os.path import join
from collections import OrderedDict
from torch.autograd import Variable

# -----------------------------------------------------------------------------

class GlobalLandmarkVisualizer:

    def __init__(self, show=False, sph_r=0.2, cmap_min=0.0, cmap_max=0.001, cmap_name='jet', token_3d='points'):
        from psbody.mesh import MeshViewer, MeshViewers
        self.viewer = MeshViewer(keepalive=True) if show else None
        self.sph_r = sph_r
        self.cmap_min = cmap_min
        self.cmap_max = cmap_max
        self.cmap_name = cmap_name
        self.token_3d = token_3d # 'spheres' or 'points'

    def save_mesh(self, mesh, save_path):
        from utils.utils import get_extension
        if get_extension(save_path) == '.obj':
            mesh.write_obj(save_path)
        elif get_extension(save_path) == '.ply':
            mesh.write_ply(save_path)
        else:
            raise RuntimeError(f'invalid save format for a mesh: {str(get_extension(save_path))}')

    def project(self, pts, K, RT):
        '''
        pts: (N,3) np.array
        K: (3,3) np.array
        RT: (3,4) np.array
        '''
        pts_cam = K.dot( RT[:,0:3].dot( pts.T ) + RT[:,3:4] )
        pts_cam = pts_cam / pts_cam[2:3, :]
        return pts_cam[0:2, :].T # (N,2))

    def show_landmarks_2d(self, lmk_pred, lmk_gt=None, K=None, RT=None, image=None, save_path=None):
        from utils.keypoint_visualizer import draw_landmarks, draw_landmarks_w_gt
        from utils.utils import value2color

        if K is None or RT is None or image is None:
            raise RuntimeError(f"show_landmarks_2d(): require K, RT and image to be non-None.")

        # projection
        lmk_pred_2d = self.project(lmk_pred, K, RT)

        if lmk_gt is None:
            lmk_gt_2d = None
            pred_colors = value2color(
                self.cmap_min * np.ones((lmk_pred.shape[0],), dtype=np.float32),
                vmin=self.cmap_min, vmax=self.cmap_max, cmap_name=self.cmap_name)  # the coldest color
            gt_color = np.array([0.4, 0.4, 0.4])  # dummy: won't be used

        else:
            lmk_gt_2d = self.project(lmk_gt, K, RT)

            # colormap error
            dist = np.linalg.norm(lmk_pred-lmk_gt, axis=1) # in 3D
            pred_colors = value2color(dist, vmin=self.cmap_min, vmax=self.cmap_max, cmap_name=self.cmap_name)
            gt_color = np.array([0.4, 0.4, 0.4])

        # draw
        canvas = draw_landmarks_w_gt(255 * image, lmk_pred_2d, lmk_gt_2d,
                                    color_pred=255 * pred_colors, color_gt=255 * gt_color, color_line=(0,255,0),
                                    thickness=3, radius=3, shift=0)
        if save_path:
            imageio.imsave(save_path, canvas.astype(np.uint8) )
        return canvas # image in range 255

    def show_landmarks_2d_mv(self, lmk_pred, lmk_gt, Ks, RTs, images, save_path=None):
        # Ks, RTs, images: list or batched array (display at each column)
        # lmk_pred: an image or a list of image; if later case, will be displayed at each row

        if isinstance(lmk_pred, list) or isinstance(lmk_gt, list):
            if isinstance(lmk_pred, list):
                level_num = len(lmk_pred)
            elif isinstance(lmk_gt, list):
                level_num = len(lmk_gt)

            canvas = []
            for lid in range(level_num):
                this_lmk_pred = lmk_pred[lid] if isinstance(lmk_pred, list) else lmk_pred
                this_lmk_gt   = lmk_gt[lid] if isinstance(lmk_gt, list) else lmk_gt
                this_canvas = [ self.show_landmarks_2d(this_lmk_pred, this_lmk_gt, Ks[idx], RTs[idx], images[idx]) for idx in range(len(Ks)) ]
                this_canvas = np.concatenate( this_canvas, axis=1 )
                canvas.append(this_canvas)
            canvas = np.concatenate( canvas, axis=0 )
        else:
            canvas = [ self.show_landmarks_2d(lmk_pred, lmk_gt, Ks[idx], RTs[idx], images[idx]) for idx in range(len(Ks)) ]
            canvas = np.concatenate( canvas, axis=1 )
        if save_path:
            imageio.imsave(save_path, canvas.astype(np.uint8) )
        return canvas # image in range 255

    def show_landmarks(self, lmk_pred, lmk_gt, algn=None, show=False, save_path=None): # alias
        return self.show_landmarks_3d(lmk_pred, lmk_gt, algn, show, save_path)

    def show_landmarks_3d(self, lmk_pred, lmk_gt, algn=None, show=False, save_path=None):
        '''
        lmk_pred: (L,3)
        lmk_gt: (L,3)
        algn: a MPI mesh instance
        '''
        # if self.token_3d in [ 'sphere', 'spheres' ]:
        #     from utils.mesh_vis_util import Sphere
        if torch.is_tensor(lmk_pred): lmk_pred = lmk_pred.detach().cpu().numpy()
        if torch.is_tensor(lmk_gt): lmk_gt = lmk_gt.detach().cpu().numpy()
        
        from utils.utils import value2color
        lmk_num = lmk_pred.shape[0]
        dist = np.linalg.norm(lmk_pred-lmk_gt, axis=1)
        colors = value2color(dist, vmin=self.cmap_min, vmax=self.cmap_max, cmap_name=self.cmap_name)
        gt_color = np.array([0.2, 0.2, 0.2])

        from psbody.mesh import Mesh
        if algn is None:
            lmk_mesh = None
        else:
            lmk_mesh = Mesh(v=algn.v, f=algn.f, vc=algn.vc if hasattr(algn, 'vc') else 0.7 * np.ones_like(algn.v))

        if self.token_3d in ['sphere', 'spheres']:
            raise NotImplementedError

        elif self.token_3d in ['point', 'points']:

            lmk_mesh_pred = Mesh(v=lmk_pred, f=None, vc=colors)
            gt_colors = np.ones_like(lmk_gt)
            gt_colors[:] = gt_color
            lmk_mesh_gt = Mesh(v=lmk_gt, f=None, vc=gt_colors)

            if lmk_mesh is None:
                lmk_mesh = lmk_mesh_pred
                lmk_mesh.concatenate_mesh(lmk_mesh_gt)
            else:
                lmk_mesh.concatenate_mesh(lmk_mesh_pred)
                lmk_mesh.concatenate_mesh(lmk_mesh_gt)

        else:
            raise RuntimeError(f"invalid self.token_3d = {self.token_3d}")

        # show
        if self.viewer:
            self.viewer.set_dynamic_meshes( [lmk_mesh] )

        if save_path:
            self.save_mesh(lmk_mesh, save_path)
        return lmk_mesh

    def show_pts_3d(self, pts, vals, save_path=None, vmin=None, vmax=None):
        '''
        pts: (L,3)
        vals: (L,)
        '''
        if torch.is_tensor(pts): pts = pts.detach().cpu().numpy()
        if torch.is_tensor(vals): vals = vals.detach().cpu().numpy()

        from utils.utils import value2color
        lmk_num = pts.shape[0]
        colors = value2color(vals, vmin=self.cmap_min if vmin is None else vmin, vmax=self.cmap_max if vmax is None else vmax, cmap_name=self.cmap_name)
        lmk_mesh = None
        for k in range( lmk_num ):

            # create sphere
            # from utils.mesh_vis_util import Sphere        
            sph_pred = Sphere(pts[k,:], self.sph_r).to_mesh(colors[k])

            if lmk_mesh is None:
                lmk_mesh = sph_pred
            else:
                lmk_mesh.concatenate_mesh(sph_pred)

        # show
        if self.viewer:
            self.viewer.set_dynamic_meshes( [lmk_mesh] )

        if save_path:
            self.save_mesh(lmk_mesh, save_path)
        return lmk_mesh

    def show_cost_volume(self, cost_vol, position_vol, save_path=None):
        '''
        cost_vol: (D, D, D), assumed normalized to sum-1 within each cost_vol[:,:,:]
        position_vol: (3, D, D, D)
        '''
        if torch.is_tensor(cost_vol): cost_vol = cost_vol.detach().cpu().numpy()
        if torch.is_tensor(position_vol): position_vol = position_vol.detach().cpu().numpy()

        # map color
        from utils.utils import value2color
        selected_cost_vol = cost_vol.ravel() # (D*D*D,)
        selected_color = value2color( selected_cost_vol, vmin=self.cmap_min, vmax=self.cmap_max, cmap_name='jet' ) # (D*D*D,3)
        vol_v = position_vol.reshape((3,-1)).T # (D*D*D,3)

        from psbody.mesh import Mesh
        cost_vol_mesh = Mesh(v=vol_v, f=None, vc=selected_color)

        if save_path:
            self.save_mesh(cost_vol_mesh, save_path)
        return cost_vol_mesh

    def show_cost_volumes(self, cost_vol, position_vol, summarize_method='concate', save_path=None):
        '''
        cost_vol: (L, D, D, D), assumed normalized to sum-1 within each cost_vol[idx, :,:,:]
        position_vol: (L, 3, D, D, D), or in special case accept (3,D,D,D) (assuming shared by all channels)
        '''
        vol_num = cost_vol.shape[0]

        if summarize_method == 'concate':
            cost_vol_mesh = None
            for vid in range(vol_num):
                this_mesh = self.show_cost_volume(cost_vol=cost_vol[vid], position_vol=position_vol[vid])
                if cost_vol_mesh is None:
                    cost_vol_mesh = this_mesh
                else:
                    cost_vol_mesh.concatenate_mesh(this_mesh)

        elif summarize_method == 'max':
            # visualize the max value across channels (L)
            # assume position_vol[l] are all the same for all l in 0 ~ L-1
            if torch.is_tensor(cost_vol): cost_vol = cost_vol.detach().cpu().numpy()
            if torch.is_tensor(position_vol): position_vol = position_vol.detach().cpu().numpy()

            cost_vol_max = np.max( cost_vol, axis=0 )

            if position_vol.ndim == 4:
                position_vol_max = position_vol # special case
            else:
                position_vol_max = position_vol[0]
            cost_vol_mesh = self.show_cost_volume(cost_vol=cost_vol_max, position_vol=position_vol_max)

        else:
            raise RuntimeError(f'invalid summarize_method = {summarize_method}')

        if save_path:
            self.save_mesh(cost_vol_mesh, save_path)
        return cost_vol_mesh