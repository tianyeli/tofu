'''
open face alignment dataset for light stage

tianye li
Please see LICENSE for the licensing information
'''
import torch
import torch.utils.data as data
import os
import copy
import math
import yaml
import numpy as np
import random
from glob import glob
from PIL import Image
import imageio
from os.path import join, basename

from utils.utils import *
# from psbody.mesh import Mesh

# -----------------------------------------------------------------------------

class FaceAlignDatasetLightstageOpen(data.Dataset):

    def __init__(
        self,
        data_root,

        # parse
        mode='test',
        split_dir=None,  # stores all the meta data (calibration, data list), assume to be at join(data_root, "metadata")

        # load: decide what order and selection for views and data instances
        load_order='deterministic',
        query_view_num=3, # "query view" same as "source view"
        dataset_size=0,   # equvalent data size to control the length of an epoch
        load_order_path='',  # special file to control the loading order (during test-time)

        # online augmentation
        scale_min=0.9, # random scaling
        scale_max=1.1,
        crop_size=(348, 256), # (w,h), random crop
        brightness_sigma=0.05 / 3.0, # random brightness perturbation

        # load option (currently not used)
        enable_load_mesh=False,
        enable_crop_mesh=False,
        enable_load_scan=False,
        enable_landmark_3d=False,        
        enable_landmark_2d=False,
        lmk_num=68,
        data_path_type='rel_path',

        # debug
        debug=False
    ):

        # load        
        self.mode = mode
        self.data_root = data_root
        if split_dir in [None, '']:
            self.split_dir = join(self.data_root, 'metadata')
        else:
            self.split_dir = split_dir
        self.data_path_type = data_path_type

        self.load_order = load_order
        self.query_view_num = query_view_num
        self.dataset_size = dataset_size  # requested dataset size
        self.load_order_path=load_order_path  # special file to control the loading order (during test-time)

        # augmentation
        self.skip_image_normalization = False
        self.scale_min = scale_min # random scaling
        self.scale_max = scale_max
        self.crop_size = crop_size # (w,h), random crop
        self.brightness_sigma = brightness_sigma # random brightness perturbation

        # normalization
        # standard values from resnet:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202
        self.mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.mean = torch.from_numpy(self.mean_np)
        self.std  = torch.from_numpy(self.std_np)

        # other modalities (currently not implemented)
        self.enable_load_mesh = enable_load_mesh
        self.enable_crop_mesh = enable_crop_mesh
        self.enable_load_scan = enable_load_scan
        self.enable_landmark_3d = enable_landmark_3d        
        self.enable_landmark_2d = enable_landmark_2d
        self.lmk_num = lmk_num
        self.data_path_type = data_path_type

        # routines
        self.load_cameras(verbose=True)
        self.load_datalist(verbose=True)

    # ----------------
    # load meta info
    # ----------------

    def load_cameras(self, verbose=False):
        """ load all camera calibration info to self.cameras, a dict: subject_name -> camera dict
        Example: self.cameras['subject_001']['cam09'] returns a dict with keys: 'name', 'K', 'R', 'T', 'height', 'width', 'resolution', 'rigid_mat', 'KRT'
        """
        # parse
        import glob
        calib_dir = join(self.split_dir, "calib")
        parsed_calib_paths = sorted(glob.glob(join(calib_dir, "subject_???.json")))
        print(f"FaceAlignDatasetLightstageOpen::load_cameras(): parsed {len(parsed_calib_paths)} cameras")

        # load
        from utils.utils import load_json
        self.cameras = {}
        for cid, cpath in enumerate(parsed_calib_paths):
            subject_name = basename(cpath)[:-5]
            subject_camera = load_json(cpath)
            self.cameras[subject_name] = subject_camera
            if verbose: print(f"FaceAlignDatasetLightstageOpen::load_cameras(): loaded camera for subject '{subject_name}' from:\n\t{cpath}")

    def load_datalist(self, verbose=False):
        """ load the list of all data: subject/expressions, image paths, etc.
        """
        from utils.utils import load_json
        self.subject_expressions = load_json(join(self.split_dir, f"{self.mode}_subject_expressions.json"))
        self.img_name_list     = load_json(join(self.split_dir, f"{self.mode}_img_name_list.json"))
        self.img_rel_path_list = load_json(join(self.split_dir, f"{self.mode}_img_rel_path_list.json"))

        # hardcoded: all possible camera names
        self.view_names = [
            'cam09', 'cam10', 'cam11', 'cam12', 'cam13', \
            'cam14', 'cam16', 'cam17', 'cam18', 'cam19', \
            'cam20', 'cam21', 'cam22', 'cam23', 'cam24'
        ]

        # existing data number
        self.available_data_num = len(self.subject_expressions)

        # the data number that we will use i.e. self.__len__
        if self.dataset_size > 0:
            self.requested_data_num = min(self.dataset_size, self.available_data_num)
        else:
            self.requested_data_num = self.available_data_num

        if verbose: print(f"FaceAlignDatasetLightstageOpen::load_datalist(): loaded data list containing {self.available_data_num} data. Requested data num is {self.requested_data_num}")

    # ----------------
    # load data
    # ----------------

    def get_data_paths(self, index):
        # assign data paths for loading

        if self.load_order == 'deterministic':
            # note: this mode will strictly load the instance and view provided by the lists
            raise NotImplementedError
            # data_idx     = self.load_data_idx[ index ]
            # ref_view_idx = self.load_ref_view_idx[ index ]
            # qry_view_idx = self.load_qry_view_idx[ index ]

        elif self.load_order == 'online_random_view':
            # note: will choose`index`-th instance, but the view selection is random

            data_idx = index
            subject, expression = self.subject_expressions[data_idx]

            # get data paths
            img_rel_paths = self.img_rel_path_list[subject][expression]  # always use the relative image path
            actual_view_num = len(img_rel_paths)

            # sample view
            np.random.seed() # online randomness
            ref_view_idx = np.random.randint(0, actual_view_num)
            avail_list = np.setdiff1d(np.arange(actual_view_num), np.array([ref_view_idx]))
            qry_view_idx = np.random.choice(avail_list, size=self.query_view_num, replace=False)

            # ref images
            ref_img_path = join(self.data_root, img_rel_paths[ref_view_idx])
            ref_mask_path = ''  # ignore for now

            # query images
            qry_img_paths = [
                join(self.data_root, img_rel_paths[qid]) for qid in qry_view_idx
            ]
            qry_mask_paths = [
                '' for qid in qry_view_idx
            ]

            # alignment
            algn_path = ''

            # view name
            img_names = self.img_name_list[subject][expression]
            ref_vw_name  = img_names[ref_view_idx][:-4]
            qry_vw_names = [
                img_names[qid][:-4] for qid in qry_view_idx
            ]

        else:
            raise RuntimeError(f"unrecognizable load_order: {self.load_order}. Expect: 'deterministic' or 'online_random_view'")

        return ref_img_path, qry_img_paths, \
               ref_mask_path, qry_mask_paths, algn_path, \
               data_idx, ref_view_idx, qry_view_idx, \
               subject, expression, ref_vw_name, qry_vw_names

    # TODO: move these two functions to other places
    def normalize_image(self, image):
        # assume image in (H,W,3) in numpy array or (B,3,H,W) in tensor
        if isinstance(image, np.ndarray):
            if image.ndim !=3 or image.shape[2] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return ( image - self.mean_np.reshape((1,1,3)) ) / self.std_np.reshape((1,1,3))
        elif torch.is_tensor(image):
            if image.ndimension() !=4 or image.shape[1] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return ( image - self.mean.view(1,3,1,1).to(image.device) ) / self.std.view(1,3,1,1).to(image.device)
        else:
            raise RuntimeError(f"unrecognizable image type {type(image)}")

    def denormalize_image(self, image):
        # assume image in (H,W,3) in numpy array or (B,3,H,W) in tensor
        if isinstance(image, np.ndarray):
            if image.ndim !=3 or image.shape[2] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return image * self.std_np.reshape((1,1,3)) + self.mean_np.reshape((1,1,3))
        elif torch.is_tensor(image):
            if image.ndimension() !=4 or image.shape[1] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return image * self.std.view(1,3,1,1).to(image.device) + self.mean.view(1,3,1,1).to(image.device)
        else:
            raise RuntimeError(f"unrecognizable image type {type(image)}")

    def read_img_with_camera(self, img_path, mask_path, subject_name, view_name):
        """ read img, mask and camera given subject and view
        geometric (random scaling and cropping) and photometric (perturbing brightness) augmentation to the image
        the camera intrinsics will also be adjusted according to the scaling and cropping
        """

        from utils.data_augment import resize_intrinsic, get_random_crop_offsets, scale_crop

        # read
        try:
            img_cur = imageio.imread(img_path).astype(np.float32) / 255.
        except:
            raise RuntimeError(f"error loading img {img_path}")

        cur_height, cur_width, _ = img_cur.shape
        mask = None # to add later
        camera = self.cameras[subject_name][view_name]

        # adjust K for image resizing
        orig_width, orig_height = camera['resolution']
        K = resize_intrinsic(np.array(camera['K'], dtype=np.float32), orig_height=orig_height, orig_width=orig_width, new_height=cur_height, new_width=cur_width)

        # geometric augmentation by random scaling and cropping
        np.random.seed()
        scale_factor = self.scale_min + (self.scale_max - self.scale_min) * np.random.random()
        h_offset, w_offset = get_random_crop_offsets(self.crop_size, height=cur_height, width=cur_width)
        img_aug, K_aug = scale_crop(img_cur, self.crop_size, h_offset, w_offset, scale_factor, K=K)

        # random brightness perturbation
        perturb = 1.0 + self.brightness_sigma * np.random.randn(1,1,3)
        img_aug = img_aug * perturb
        img_aug = np.clip(img_aug, 0., 1.)

        # # debug
        # print( f"scale_factor = {scale_factor}, crop offsets (h,w) = {h_offset},{w_offset}, perturb={perturb.ravel()} (cur size (h,w) = {cur_height}, {cur_width})" )
        # print( f"K_aug = {K_aug}" )
        # import ipdb; ipdb.set_trace()

        # normalize rgb
        if self.skip_image_normalization:
            pass
        else:
            img_aug = self.normalize_image(img_aug)

        # to tensor
        img_aug = torch.FloatTensor(torch.from_numpy(img_aug.astype(np.float32))).permute(2,0,1).contiguous() # (3,H,W) range (0,1) only rgb
        mask = None
        K_aug = torch.FloatTensor(torch.from_numpy(K_aug.astype(np.float32)))
        RT = torch.FloatTensor(torch.from_numpy(np.array(camera['rigid_mat'], dtype=np.float32).copy()))
        return img_aug, mask, K_aug, RT

    def read(self, index):
        """ get a data instance
        :param index: data instance index within 0 to __len__
        :return: a dictionary of all images, camera matrices, meta info and optionally meshes
        """

        # enable_profiling = False

        # import time
        # if enable_profiling:
        #     tic = time.time()

        # get paths
        ref_img_path, qry_img_paths, \
            ref_mask_path, qry_mask_paths, algn_path, \
            data_idx, ref_view_idx, qry_view_idx, \
            subj, fr_name, ref_vw_name, qry_vw_names = self.get_data_paths(index)

        # if enable_profiling:
        #     toc_paths = time.time()

        # load images and cameras
        ref_img, ref_mask, ref_K, ref_RT = self.read_img_with_camera(img_path=ref_img_path, mask_path=ref_mask_path,
                                                                     subject_name=subj, view_name=ref_vw_name)

        qry_imgs, qry_masks, qry_Ks, qry_RTs = [], [], [], []
        for ip, mp, vn in zip( qry_img_paths, qry_mask_paths, qry_vw_names ):

            ii, mm, kk, rt = self.read_img_with_camera(img_path=ip, mask_path=mp,
                                                       subject_name=subj, view_name=vn)

            qry_imgs.append(ii.unsqueeze(0))
            # qry_masks.append( mm.unsqueeze(0) )
            qry_Ks.append(kk.unsqueeze(0))
            qry_RTs.append(rt.unsqueeze(0))

        # concate
        qry_imgs = torch.cat( qry_imgs, dim=0 )
        # qry_masks = torch.cat( qry_masks, dim=0 )
        qry_Ks = torch.cat( qry_Ks, dim=0 )
        qry_RTs = torch.cat( qry_RTs, dim=0 )

        # if enable_profiling:
        #     toc_imgs_cams = time.time()

        # ----------------------------------

        # # load alignment mesh
        if self.enable_load_mesh:
            # provides: algn_path, algn_v, algn_f
            # if self.enable_crop_mesh, then it will crop the mesh (up to the defined mask)
            raise NotImplementedError

        # landmarks
        if self.enable_landmark_3d:
            # provides: lmk_3d
            # if self.enable_landmark_2d is True, provides ref_lmk_2d, qry_lmks_2d
            raise NotImplementedError

            # landmarks 2D
            if self.enable_landmark_2d:
                raise NotImplementedError

        # scan
        # note: scans have different num of points for different meshes
        if self.enable_load_scan:
            # provides: scan_path, scan_v, scan_f
            raise NotImplementedError

        # toc_lmks = time.time()
        # if enable_profiling:
        #     print(f"\tpaths: {toc_paths - tic:3.2f}")
        #     print(f"\timgs, cams: {toc_imgs_cams - toc_paths:3.2f}")
        #     print(f"\tlmk: {toc_lmks - toc_imgs_cams:3.2f}\n")

        # print(f"FaceAlignDatasetLightstageOpen::read(): loaded data: {subj} - {fr_name} - {ref_vw_name} - {', '.join(qry_vw_names)}")

        # ----------------------------------

        # return dict
        data = {
            # img
            'ref_img': ref_img,   # (3,H,W)
            'qry_imgs': qry_imgs, # (Q,3,H,W)
            # 'ref_mask': ref_mask,   # (H,W)
            # 'qry_masks': qry_masks, # (Q,H,W)

            # cam
            'ref_K': ref_K,     # (3,3)
            'qry_Ks': qry_Ks,   # (Q,3,3)
            'ref_RT': ref_RT,   # (3,4)
            'qry_RTs': qry_RTs, # (Q,3,4)

            # meta
            'index': index,
            'data_idx': data_idx,
            'ref_view_idx': ref_view_idx,
            'qry_view_idx': qry_view_idx,
            'ref_img_path': ref_img_path,
            'qry_img_paths': qry_img_paths,
            'ref_mask_path': ref_mask_path,
            'qry_mask_paths': qry_mask_paths,
        }
        if self.enable_load_mesh:
            data['algn_v'] = algn_v # (N,3) or None
            data['algn_f'] = algn_f # (Nf,3) or None
            data['algn_path'] = algn_path #  str

        if self.enable_load_scan:
            data['scan_v'] = scan_v # (N',3) or None
            data['scan_f'] = scan_f # (Nf',3) or None
            data['scan_path'] = scan_path #  str

        if self.enable_landmark_3d:
            data['lmk_3d'] = lmk_3d  # (L,3), dlib: L=68 or other num
            if self.enable_landmark_2d:
                data['ref_lmk_2d'] = ref_lmk_2d # (L,2) or None
                data['qry_lmks_2d'] = qry_lmks_2d # (Q,L,2) or None

        return data

    def __len__(self):
        return self.requested_data_num

    def __getitem__(self, index):
        return self.read(index)