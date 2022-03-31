'''
mesh resampler
tianye li
Please see LICENSE for the licensing information
'''
import math
import heapq
import numpy as np
import scipy.sparse as sp
from psbody.mesh import Mesh

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------

def scipy_to_torch_sparse(scp_matrix):
    import torch
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

# # -----------------------------------------------------------------------------

def vertex_normals(vertices, faces):
    """
    computer vertex normal vectors, from SoftRas
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

# -----------------------------------------------------------------------------

def sparse_bmm(sparse, dense):
    # sparse: (M, N)
    # dense: (B, N, P)
    # https://github.com/pytorch/pytorch/issues/14489#issuecomment-459827637
    import torch

    # version 1
    return torch.stack([sparse.mm(dd) for dd in dense])

    # # version 2
    # M = sparse.shape[0]
    # B, N, P = dense.shape
    # vectors = dense.transpose(0,1).reshape(N, -1)
    # return sparse.mm(vectors).reshape(M, B, P).transpose(0, 1)

# -----------------------------------------------------------------------------

class MeshResampler(nn.Module):

    def __init__(
        self,
        info_path,
        level_start=None,
        level_end=None,
        sample_num=-1,
        sample_base_threshold=0.3,
        sample_propogate_threshold=0.5,
        normal_scale_factor=1.0
    ):
        super(MeshResampler, self).__init__()
        from utils.utils import get_extension
        self.info_path = info_path
        ext = get_extension(info_path)
        if ext in ['.obj', '.ply']:
            self.compute_mesh_resample_info(info_path)
        elif ext in ['.pkl']:
            self.load_mesh_resample_info(info_path)
        else:
            raise RuntimeError(f"invalid extension name of {info_path}")

        # level = 0 is the highest resolution
        self.level_start = self.level_num if level_start is None else level_start
        self.level_end = 0 if level_end is None else level_end
        self.level_process = self.level_start - self.level_end

        # settings for generating sampling masks
        if sample_num <= 0:
            print(f"MeshResampler::__init__(): not using patch sampling")
        else:
            print(f"MeshResampler::__init__(): using patch sampling")
            print(f"MeshResampler::__init__(): sample_num = {sample_num}")
            print(f"MeshResampler::__init__(): sample_base_threshold = {sample_base_threshold}")
            print(f"MeshResampler::__init__(): sample_propogate_threshold = {sample_propogate_threshold}")
        self.sample_num = sample_num
        self.sample_base_threshold = sample_base_threshold
        self.sample_propogate_threshold = sample_propogate_threshold

        # ict template resampler use centimeters
        # so e.g. for flame meshes (in meters), then normal_scale_factor should be 0.01
        self.normal_scale_factor = normal_scale_factor
        print(f"MeshResampler::__init__(): normal_scale_factor = {normal_scale_factor}")

    def load_mesh_resample_info(self, file_path, verbose=False):
        '''load info from processed data
        '''
        from utils.utils import load_binary_pickle
        info = load_binary_pickle(file_path)
        self.templates = info['M']
        self.num_nodes = info['num_nodes']
        self.v2l = { vn: idx for idx, vn in enumerate(self.num_nodes) }
        self.level_num = len(info['U'])

        self.vertices_templ = [torch.from_numpy(mm.v.astype(np.float32)) for mm in info['M']]
        self.faces = [torch.from_numpy(mm.f.astype(np.int64)) for mm in info['M']]
        self.vts = [torch.from_numpy(vt.astype(np.float32)) for vt in info['VT']]
        self.weights = [torch.from_numpy(vw.astype(np.float32)) for vw in info['VW']]

        if info['U'][0].shape[1] == self.vertices_templ[1].shape[0]:
            self.use_normals = False
        else:
            self.use_normals = True
            print(f" ---- Surprise! Use normal to upsample mesh ----")

        for level in range(self.level_num):
            self.register_buffer(f'D_{level}', scipy_to_torch_sparse(info['D'][level]))
            self.register_buffer(f'U_{level}', scipy_to_torch_sparse(info['U'][level]))
            self.register_buffer(f'A_{level}', scipy_to_torch_sparse(info['A'][level]))
        if verbose: print(f"loaded mesh resample info from {file_path}")

    def compute_mesh_resample_info(self, file_path, save_path=None, verbose=False):
        '''compute info if provided template mesh path
        '''
        raise RuntimeError(f"not fully tested")

        if verbose: print(f"computing mesh resample info from mesh: {file_path}")
        template_mesh = Mesh(filename=file_path)

        if verbose: print('Generating transforms')
        M, A, D, U = generate_transform_matrices(template_mesh, factors=[4, 4, 4, 4])

        info = {
            'M': M,
            'A': A,
            'D': D,
            'U': U,
            'num_nodes': [len(M[i].v) for i in range(len(M))],
            'faces': [mm.f for mm in M],
        }
        # note: missing vt

        # save
        if save_path is not None:
            from utils.utils import save_binary_pickle
            save_binary_pickle(info, save_path)
            print(f"resample info saved at: {save_path}")

        # # debug load
        # from utils.utils import load_binary_pickle
        # db_data = load_binary_pickle(save_path)

        # set
        self.templates = info['M']
        self.num_nodes = info['num_nodes']
        self.v2l = { vn: idx for idx, vn in enumerate(self.num_nodes) }
        self.level_num = len(info['U'])
        self.faces = [torch.from_numpy(mm.f.astype(np.int64)) for mm in info['M']]
        for level in range(self.level_num):
            self.register_buffer(f'D_{level}', scipy_to_torch_sparse(info['D'][level]))
            self.register_buffer(f'U_{level}', scipy_to_torch_sparse(info['U'][level]))
            self.register_buffer(f'A_{level}', scipy_to_torch_sparse(info['A'][level]))
        if verbose: print(f"computed mesh resample info from {file_path}")

    def print_setting(self):

        print(f"Available resampling matrices:")
        for level_idx in range(self.level_num):
            print(f"\t\t-level {level_idx}/{self.level_num}:\t"
                f"D: {self._D(level_idx).shape}\t"
                f"U: {self._U(level_idx).shape}\t"
                f"A: {self._A(level_idx).shape}\t")

        print(f"\tWill use to sample:")
        print(f"\t\t-self.level_start = {self.level_start}")
        print(f"\t\t-self.level_end = {self.level_end}")
        print(f"\t\t-self.level_process = {self.level_process}")

        for idx in range(0, self.level_process):
            level_idx = self.level_end + idx
            print(f"\t\t-level {level_idx}/{self.level_num}:\t"
                f"D: {self._D(level_idx).shape}\t"
                f"U: {self._U(level_idx).shape}\t"
                f"A: {self._A(level_idx).shape}\t")

    def _D(self, level):
        return getattr(self, f"D_{level}")

    def _U(self, level):
        return getattr(self, f"U_{level}")

    def _A(self, level):
        return getattr(self, f"A_{level}")

    def get_level(self, vertices):
        pn = vertices.shape[1]
        if pn in self.v2l.keys():
            return self.v2l[pn]
        else:
            raise RuntimeError(f"input vertices number {pn} doesn't match to any level")

    def get_faces(self, vertices):
        return self.faces[self.get_level(vertices)]

    def get_weights(self, vertices):
        return self.weights[self.get_level(vertices)]

    def get_init(self, vertices_full):
        ''' prepare points initialization (as input)
        '''
        assert self.get_level(vertices_full) == 0, f"input should be highest resolution (level 0)"
        return self.downsample_by(vertices_full, by=self.level_start), \
               torch.from_numpy(self.templates[self.level_start].f.astype(np.int64))[None, :].repeat(vertices_full.shape[0], 1, 1)

    def get_gt(self, vertices_full):
        ''' prepare point ground truth
        '''
        assert self.get_level(vertices_full) == 0, f"input should be highest resolution (level 0)"
        return self.downsample_by(vertices_full, by=self.level_end), \
               torch.from_numpy(self.templates[self.level_end].f.astype(np.int64))[None, :].repeat(vertices_full.shape[0], 1, 1)

    def get_gt_list(self, vertices_full):
        ''' prepare list of point ground truth
        Args:
            vertices_full: batched vertices in highest resolution, (B, N, 3)
        Returns:
            vv, ff: list of vertices and faces, with [lowest resol -> highest resol], level in range of requested levels
        '''
        assert self.get_level(vertices_full) == 0, f"input should be highest resolution (level 0)"
        bs = vertices_full.shape[0]
        cur_v = vertices_full

        vv, ff = [], []
        vv_high, ff_high = self.get_gt(vertices_full)
        vv.append(vv_high)
        ff.append(ff_high)

        cur_v = vv_high
        for idx in range(1, self.level_process):
            cur_v = self.downsample(cur_v)
            vv.append(cur_v)
            ff.append(torch.from_numpy(self.templates[self.level_end+idx].f.astype(np.int64))[None, :].repeat(bs, 1, 1))

        return vv[::-1], ff[::-1]

    def get_templates(self):
        vv, ff = [], []
        for idx in range(0, self.level_process):
            vv.append(self.templates[self.level_end+idx].v.astype(np.float32))
            ff.append(self.templates[self.level_end+idx].f.astype(np.int64))
        return vv[::-1], ff[::-1]

    def get_random_mask_base(self, sample_num=None, use_weight_mask=False, threshold=None):
        """ returns random selection mask (of 0s and 1s), in shape (v_num,)
        """
        v_num = self.num_nodes[self.level_start]

        if sample_num is None:
            sample_num = self.sample_num

        if sample_num > 0:
            # random sample
            if threshold is None:
                threshold = self.sample_base_threshold

            if use_weight_mask:
                this_weight = self.weights[self.level_start]
                pool = np.nonzero(this_weight.detach().cpu().numpy() > threshold)[0]
            else:
                pool = np.arange(v_num)  # all is possible

            if sample_num > pool.shape[0]:
                print(f"Warning: sample_num ({sample_num}) > pool size ({pool.shape[0]}). Enforcing it to be same.")
                sample_num = pool.shape[0]

            # hack
            np.random.seed(0)
            print(f"HACK: set random seed 0 for patch mask")

            sample_idx = np.random.choice(pool, size=sample_num, replace=False)
            sample_mask = torch.zeros(v_num,)
            sample_mask[sample_idx] = 1

        else:
            # as if no sampling is done
            sample_mask = torch.ones(v_num,)

        return sample_mask

    def get_random_masks_list(self, vertices_full, base_mask, threshold=None):
        """ get masks that randomly samples points, but keep correspondence among the levels
        the list elements are in shape (N_i,), N_i starts from start_level-1 and ends at end level, inclusively
        """
        bs = vertices_full.shape[0]
        device = vertices_full.device

        if threshold is None:
            threshold = self.sample_propogate_threshold

        # base
        assert base_mask.shape[0] == self.num_nodes[self.level_start], \
            f"assume base_mask ({base_mask.shape[0]}) has the same shape of the same level vertice num ({self.num_nodes[self.level_start]})"
        base_mask = base_mask.to(device)

        def _upsample_mask(base):
            _base = base[None, :, None].repeat(1, 1, 3) # (1, N, 3)
            up_mask = self.upsample(_base, no_normals=True)
            up_mask = (up_mask[0, :, 0] > threshold).float() # (N_higher,)
            return up_mask

        rm = []
        new_mask = base_mask
        for idx in range(0, self.level_process):
            new_mask = _upsample_mask(new_mask)
            rm.append(new_mask)

        return rm, base_mask

    def get_weights_list(self, vertices_full):
        ''' prepare list of point weights
        '''
        bs = vertices_full.shape[0]
        device = vertices_full.device
        vw = []
        for idx in range(0, self.level_process):
            vw.append(self.weights[self.level_end+idx][None, :].repeat(bs, 1).to(device))
        # for idx in range(self.level_start, self.level_end-1, -1):
        return vw[::-1]

    def upsample(self, vertices, no_normals=False):
        ''' upsample vertices by 1 level
        Args:
            vertices: tensor (B, N, 3)
        '''
        level = self.get_level(vertices)
        if level == 0:
            import warnings
            warnings.warn(f"vertices at highest level. No upsampling is applied")
            return vertices
        else:
            U = self._U(level-1)
            if self.use_normals and U.shape[1] == 2 * vertices.shape[1]:
                # from modules.vertex_normals import vertex_normals
                bs = vertices.shape[0]
                device = vertices.device
                faces = self.get_faces(vertices)[None, :, :].repeat(bs, 1, 1).to(device)
                if no_normals:
                    normals = torch.zeros_like(vertices).to(device)
                else:
                    normals = vertex_normals(vertices, faces) * self.normal_scale_factor
                return sparse_bmm(U, torch.cat((vertices, normals), dim=1))
            else:
                return sparse_bmm(U, vertices)

    def downsample(self, vertices, no_normals=False):
        ''' downsample vertices by 1 level
        Args:
            vertices: tensor (B, N, 3)
        '''
        level = self.get_level(vertices)
        if level == self.level_num:
            import warnings
            warnings.warn(f"vertices at lowest level. No downsampling is applied")
            return vertices
        else:
            D = self._D(level)
            if self.use_normals and D.shape[1] == 2 * vertices.shape[1]:
                # from modules.vertex_normals import vertex_normals
                bs = vertices.shape[0]
                device = vertices.device
                faces = self.get_faces(vertices)[None, :, :].repeat(bs, 1, 1).to(device)
                if no_normals:
                    normals = torch.zeros_like(vertices).to(device)
                else:
                    normals = vertex_normals(vertices, faces) * self.normal_scale_factor
                return sparse_bmm(D, torch.cat((vertices, normals), dim=1))
            else:
                return sparse_bmm(D, vertices)

    def upsample_by(self, vertices, by=1, no_normals=False):
        ''' upsample vertices by "by" level
        Args:
            vertices: tensor (B, N, 3)
            by: int
        '''
        for bid in range(by):
            vertices = self.upsample(vertices, no_normals=no_normals)
        return vertices

    def downsample_by(self, vertices, by=1):
        ''' downsample vertices by "by" level
        Args:
            vertices: tensor (B, N, 3)
            by: int
        '''
        for bid in range(by):
            vertices = self.downsample(vertices)
        return vertices