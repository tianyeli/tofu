'''
alignment visualizer

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

scan_color = [ 224., 177., 189. ]
algn_color = [ 155., 184., 143. ]

# -----------------------------------------------------------------------------

class AlignmentVisualizer(object):

    def __init__(self, cmap_min=0.0, cmap_max=0.001, cmap_name='jet', show=False, color_scheme='default'):
        from psbody.mesh import MeshViewer, MeshViewers
        self.viewer = MeshViewer(keepalive=True) if show else None
        self.cmap_min = cmap_min
        self.cmap_max = cmap_max
        self.cmap_name = cmap_name
        self.valid_metrics = ['s2m', 'm2s', 'v2v']

        # color scheme
        if color_scheme == 'default':
            self.scan_color = scan_color
            self.algn_color = algn_color
        elif color_scheme == 'gray_blue':
            self.scan_color = [199, 199, 199]
            # self.algn_color = [27, 125, 203]  # v0 jiayi
            # self.algn_color = [137, 209, 254]
            self.algn_color = [172, 222, 254]  # v2 chosen
            # self.algn_color = [155, 203, 255]
            # self.algn_color = [184, 210, 242]
        elif color_scheme == 'gray_gray':
            self.scan_color = [199, 199, 199]
            self.algn_color = [199, 199, 199]
        else:
            raise RuntimeError(f"invalid color_scheme = {color_scheme}")
        print(f"AlignmentVisualizer::__init__(): applied color_scheme: {color_scheme}")

    def save_mesh(self, mesh, save_path):
        from utils.utils import get_extension
        if get_extension(save_path) == '.obj':
            mesh.write_obj(save_path)
        elif get_extension(save_path) == '.ply':
            mesh.write_ply(save_path)
        else:
            raise RuntimeError(f'invalid save format for a mesh: {str(get_extension(save_path))}')

    # ---- metrics and color ----

    def compute_s2m(self, scan, mesh):
        tree = mesh.compute_aabb_tree()
        matched_f_idx, matched_pts = tree.nearest(scan.v)
        dist = np.linalg.norm( matched_pts - scan.v, axis=1 )
        return dist

    def compute_m2s(self, scan, mesh):
        tree = scan.compute_aabb_tree()
        matched_f_idx, matched_pts = tree.nearest(mesh.v)
        dist = np.linalg.norm( matched_pts - mesh.v, axis=1 )
        return dist

    def compute_v2v(self, vert_1, vert_2):
        return np.linalg.norm( vert_1 - vert_2, axis=1 )

    def set_vc(self, vert, color, bg_color=[0.5, 0.5, 0.5], mask=None):
        vc = np.ones_like(vert)
        if mask is None:
            vc[:] = np.asarray(color).ravel()
        else:
            vc[:] = np.asarray(bg_color).ravel()
            vc[mask>0] = np.asarray(color).ravel()
        if vc.max() > 1.0: vc = vc / 255.
        return vc

    # ---- map mesh for visualization ----

    def map_algn(self, mesh):
        from psbody.mesh import Mesh
        return Mesh(v=mesh.v, f=mesh.f, vc=self.set_vc(mesh.v, self.algn_color))

    def map_scan(self, scan):
        from psbody.mesh import Mesh
        return Mesh(v=scan.v, f=scan.f, vc=self.set_vc(scan.v, self.scan_color))

    def map_s2m(self, scan, mesh):
        from psbody.mesh import Mesh
        from utils.utils import value2color
        dist = self.compute_s2m(scan=scan, mesh=mesh)
        vc = value2color(dist, vmin=self.cmap_min, vmax=self.cmap_max, cmap_name=self.cmap_name)
        return Mesh(v=scan.v, f=scan.f, vc=vc), dist

    def map_m2s(self, scan, mesh):
        from psbody.mesh import Mesh
        from utils.utils import value2color
        dist = self.compute_m2s(scan=scan, mesh=mesh)
        vc = value2color(dist, vmin=self.cmap_min, vmax=self.cmap_max, cmap_name=self.cmap_name)
        return Mesh(v=mesh.v, f=mesh.f, vc=vc), dist

    def map_v2v(self, scan, mesh):
        from utils.utils import value2color
        from psbody.mesh import Mesh
        if isinstance(scan, np.ndarray) and isinstance(mesh, np.ndarray):
            dist = self.compute_v2v(scan, mesh)
            vc = value2color(dist, vmin=self.cmap_min, vmax=self.cmap_max, cmap_name=self.cmap_name)
            return vc, dist
        elif isinstance(scan, Mesh) and isinstance(mesh, Mesh):
            dist = self.compute_v2v(scan.v, mesh.v)
            vc = value2color(dist, vmin=self.cmap_min, vmax=self.cmap_max, cmap_name=self.cmap_name)
            return Mesh(v=scan.v, f=scan.f, vc=vc), dist
        else:
            raise RuntimeError(f"input valid type")

    def visualize_compare(self, scan, mesh, metrics=['s2m'], return_meshes=False, return_metrics=False):
        # assuming the scan here is gt registered mesh, which has the same number of vertices as input mesh
        from psbody.mesh import Mesh
        vis_meshes = []

        # colormap the meshes
        vis_scan = self.map_scan(scan)
        vis_mesh = self.map_algn(mesh)
        vis_overlay = Mesh(v=vis_scan.v, f=vis_scan.f, vc=vis_scan.vc).concatenate_mesh(vis_mesh)
        vis_meshes += [vis_scan, vis_mesh, vis_overlay]

        # compute and colormap the metrics
        vis_metrics = []
        for mt in metrics:
            if mt in self.valid_metrics:
                try:
                    this_mesh, this_metric = getattr(self, f'map_{mt}')(scan, mesh)
                    vis_meshes.append(this_mesh)
                    vis_metrics.append(this_metric)
                except:
                    pass

        if return_meshes:
            if return_metrics:
                return vis_meshes, vis_metrics
            else:
                return vis_meshes
        else:
            if self.viewer is None:
                from psbody.mesh import MeshViewer, MeshViewers        
                self.viewer = MeshViewers(
                    shape=(1, len(vis_meshes)), titlebar="Compare Scan, Align, Overlay and Metrics",
                    keepalive=False, window_width=400, window_height=400)
            for wid, this_mesh in enumerate(vis_meshes):
                self.viewer[0][wid].set_dynamic_meshes([this_mesh])

    def visualize_compare_hardcoded(self, gt_scan, gt_mesh, mesh):
        # hardcoded scripts: always return mesh and metrics
        from psbody.mesh import Mesh

        # colormap the meshes
        vis_gt_scan = self.map_scan(gt_scan)
        vis_gt_mesh = self.map_scan(gt_mesh)
        vis_mesh = self.map_algn(mesh)
        vis_overlay = Mesh(v=vis_gt_scan.v, f=vis_gt_scan.f, vc=vis_gt_scan.vc).concatenate_mesh(vis_mesh)

        vis_s2m, met_s2m = self.map_s2m(gt_scan, mesh)
        vis_m2s, met_m2s = self.map_m2s(gt_scan, mesh)
        vis_v2v, met_v2v = self.map_v2v(gt_mesh, mesh)

        vis_meshes = [vis_gt_scan, vis_gt_mesh, vis_mesh, vis_overlay, vis_s2m, vis_m2s, vis_v2v]
        vis_metrics = [met_s2m, met_m2s, met_v2v]
        return vis_meshes, vis_metrics

    def visualize_compare_w_lmk(self, scan, mesh, scan_lmk, mesh_lmk,
        mask_scan_lmk=None, mask_mesh_lmk=None, sph_r=0.001, return_meshes=False, always_new_window=False):
        from psbody.mesh import Mesh
        from utils.mesh_vis_util import visualize_mesh_w_3D_landmarks as map_lmk

        vis_scan = self.map_scan(scan)
        vis_mesh = self.map_algn(mesh)
        vis_scan_lmk = map_lmk(None, scan_lmk, sph_r=sph_r, landmarks_color=self.set_vc(scan_lmk, [1.0, 0.5, 0.5], mask=mask_scan_lmk), return_mesh=True)
        vis_mesh_lmk = map_lmk(None, mesh_lmk, sph_r=sph_r, landmarks_color=self.set_vc(mesh_lmk, [0., 1.0, 0.], mask=mask_mesh_lmk), return_mesh=True)
        vis_scan_combined = vis_scan_lmk.concatenate_mesh(vis_scan)
        vis_mesh_combined = vis_mesh_lmk.concatenate_mesh(vis_mesh)

        # lmk_err, _ = self.map_v2v(scan_lmk, mesh_lmk)
        vis_overlay = Mesh(v=vis_scan_combined.v, f=vis_scan_combined.f, vc=vis_scan_combined.vc).concatenate_mesh(vis_mesh_combined)

        vis_s2m, _ = self.map_s2m(scan, mesh)

        if scan.v.shape[0] == mesh.v.shape[0]:
            enable_v2v = True
            vc_v2v, _ = self.map_v2v(scan.v, mesh.v)
            vis_v2v = Mesh(v=scan.v, f=scan.f, vc=vc_v2v)
            window_num = 5

        else:
            enable_v2v = False
            window_num = 5 # 4  --- hack

        if return_meshes:
            if enable_v2v:
                return vis_scan_combined, vis_mesh_combined, vis_overlay, vis_s2m, vis_v2v
            else:
                return vis_scan_combined, vis_mesh_combined, vis_overlay, vis_s2m

        else:
            if always_new_window:
                from psbody.mesh import MeshViewer, MeshViewers        
                viewer = MeshViewers(
                    shape=(1, window_num), titlebar="Compare Align vs Scan",
                    keepalive=True, window_width=400, window_height=400)
                viewer[0][0].set_dynamic_meshes([vis_scan_combined])
                viewer[0][1].set_dynamic_meshes([vis_mesh_combined])
                viewer[0][2].set_dynamic_meshes([vis_overlay])
                viewer[0][3].set_dynamic_meshes([vis_s2m])
                if enable_v2v: viewer[0][4].set_dynamic_meshes([vis_v2v])
                return viewer

            else:
                if self.viewer is None:
                    from psbody.mesh import MeshViewer, MeshViewers        
                    self.viewer =  MeshViewers(
                        shape=(1, window_num), titlebar="Compare Align vs Scan",
                        keepalive=False, window_width=400, window_height=400)
                self.viewer[0][0].set_dynamic_meshes([vis_scan_combined])
                self.viewer[0][1].set_dynamic_meshes([vis_mesh_combined])
                self.viewer[0][2].set_dynamic_meshes([vis_overlay])
                self.viewer[0][3].set_dynamic_meshes([vis_s2m])
                if enable_v2v: self.viewer[0][4].set_dynamic_meshes([vis_v2v])

# ----------------------------------------------------------------------------------------------

# def test():

#     from psbody.mesh import Mesh
#     vis = AlignmentVisualizer(cmap_min=0.0, cmap_max=0.01, cmap_name='jet', show=False)

#     scan_path = '/home/tli/Dropbox/DenseFaceTracking/meta/transfer_topology/ict_vs_bvmodel/LS2BV/standard_scaled.obj'
#     mesh_path = '/home/tli/Dropbox/DenseFaceTracking/meta/transfer_topology/ict_vs_bvmodel/LS2BV/faceX_0.1.obj'
#     scan = Mesh(filename=scan_path)
#     mesh = Mesh(filename=mesh_path)
#     meshes, metrics = vis.visualize_compare(scan, mesh, metrics=['s2m', 'm2s', 'v2v'], return_meshes=True, return_metrics=True)

#     import ipdb; ipdb.set_trace()

# # ----------------------------------------------------------------------------------------------

# if __name__ == '__main__':

#     test()