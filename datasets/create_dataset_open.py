'''
create dataset for train/test

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

def create_dataset(args, mode='train', enforce_test_setting=False):

    # choose the appropriate data size
    if mode == 'train':
        dataset_size = args.train_data_size
    elif mode == 'test':
        if enforce_test_setting:
            dataset_size = args.test_data_size
        else:
            dataset_size = args.val_data_size

    dataset = None

    if args.dataset_id in ['', 'lightstage']:  # default dataset id
        # define `dataset` instance of a Dataset
        # example:
        # from datasets.face_align_dataset_lightstage import FaceAlignDatasetLightstage
        # dataset = FaceAlignDatasetLightstage(...)

        if enforce_test_setting:
            # load_order = 'deterministic'  # TODO: should be deterministic -- will change later
            load_order = 'online_random_view'

            # enforce no geometric or photometic augmentation
            scale_min = 1.0  # no random scaling
            scale_max = 1.0  # no random scaling
            crop_size = (512, 376)  # (w,h), no random crop
            brightness_sigma = 0.0 / 3.0 # no random brightness perturbation

        else:
            raise NotImplementedError
            # training setting
            load_order = 'online_random'

            # augmentation
            scale_min = args.scale_range[0] # random scaling
            scale_max = args.scale_range[1]
            crop_size = tuple(args.crop_size) # (w,h), random crop
            brightness_sigma = args.brightness_sigma # random brightness perturbation

        # define `dataset` instance of a Dataset
        # example:
        # from datasets.face_align_dataset_lightstage import FaceAlignDatasetLightstage
        # dataset = FaceAlignDatasetLightstage(...)
        from datasets.face_align_dataset_lightstage_open import FaceAlignDatasetLightstageOpen
        dataset = FaceAlignDatasetLightstageOpen(
            data_root=args.dataset_directory,
            mode=mode,
            split_dir=args.split_directory,
            load_order=load_order,
            query_view_num=args.query_view_num,  # view_num = query_view_num + 1
            dataset_size=dataset_size,   # equvalent data size to control the length of an epoch
            load_order_path=args.load_order_path if hasattr(args, "load_order_path") else "",
            # online augmentation
            scale_min=scale_min, # random scaling
            scale_max=scale_max,
            crop_size=crop_size, # (w,h), random crop
            brightness_sigma=brightness_sigma, # random brightness perturbation
            # load option
            enable_load_mesh=args.enable_load_mesh,
            enable_crop_mesh=True,
            enable_load_scan=False,  # not important
            enable_landmark_2d=False,  # not important
            lmk_num=68,  # not important
            data_path_type='rel_path',  # not important
            # debug
            debug=False
        )

    elif args.dataset_id == 'coma_voca':
        # define `dataset` instance of a Dataset
        # example:
        # from datasets.face_align_dataset_coma_voca import FaceAlignDatasetCOMAVOCA
        # dataset = FaceAlignDatasetCOMAVOCA(...)
        pass

    else:
        raise RuntimeError(f'invalid dataset_id = {args.dataset_id}')

    print(f"created dataset '{args.dataset_id}' for mode '{mode}'")
    return dataset

