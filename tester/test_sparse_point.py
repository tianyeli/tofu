'''
test the global stage of ToFu base
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

from utils.utils import safe_mkdir, save_binary_pickle, save_npy
from option_handler.base_test_options import BaseTestOptions
from base_tester import BaseTester

# -----------------------------------------------------------------------------

class TestOptions(BaseTestOptions):
    def initialize(self):
        BaseTestOptions.initialize(self)
        self.isTrain = False
        return self.parser

    def initialize_extra(self):
        # define specific parameters to this experiment

        # data
        self.add_arg( cate='data', abbr='pd', name='processed-directory', type=str, default='') # "save_root"
        self.add_arg( cate='data', abbr='qvn', name='query-view-num', type=int, default=3) # the total view used is qvn + 1 (ref)
        self.add_arg( cate='data', abbr='ln', name='lmk-num', type=int, default=68)

        # model
        self.add_arg( cate='model', abbr='desc-dim', name='descriptor-dim', type=int, default=32)
        self.add_arg( cate='model', abbr='feat-arch', name='feature-arch', type=str, default='resnet')
        self.add_arg( cate='model', abbr='global-arch', name='global-arch', type=str, default='v2v')
        self.add_arg( cate='model', abbr='local-arch', name='local-arch', type=str, default='v2v')
        self.add_arg( cate='model', abbr='pretr-path', name='pretrained-path', type=str, default='')

        # volumetric sparse point net
        self.add_arg( cate='model', abbr='gvd', name='global-voxel-dim', type=int, default=32)
        self.add_arg( cate='model', abbr='gvi', name='global-voxel-inc', type=float, default=1.0)
        self.add_arg( cate='model', abbr='go', name='global-origin', type=list, default=[0.0, 0.0, 0.5])
        self.add_arg( cate='model', abbr='nm', name='norm', type=str, default="bn")
        self.add_arg( cate='model', abbr='ep', name='enable-profiling', type=bool, default=False)

        # visualization
        self.add_arg( cate='vis', abbr='sphr', name='vis-sphr', type=float, default=0.2)
        self.add_arg( cate='vis', abbr='cmap-max', name='vis-cmap-max', type=float, default=0.5)

# -----------------------------------------------------------------------------

class Tester(BaseTester):

    def __init__(self, args):
        super().__init__(args)
        self.args = args

    # ---- model ----

    def register_model(self):
        # define the model
        import models.model_aligner.prototypes.model_sparse_point_global_only as models
        model = models.Model(args=self.args)
        model.print_setting()

        self.model = model
        self.load_model()

        # enforce the experiment name (the folder name for test)
        self.experiment_name = f"{self.args.experiment_id}_{self.model_name}_{1+args.query_view_num}_views"

        self.model = torch.nn.DataParallel(self.model).cuda()
        # self.model.module.print_network()
        self.set_eval()

    # ---- datasets ----

    def register_dataset(self):
        # define the dataset and dataloader
        from datasets.create_dataset_open import create_dataset
        self.dataset_test = create_dataset(self.args, mode='test', enforce_test_setting=True)
        self.dataloader_test = self.make_data_loader(self.dataset_test, cuda=True, shuffle=False)  # no shuffle for testing

    # ---- forward ----

    def feed_data(self, data):
        # receive a data instance from data loader to reorganize for model.forward format
        # records to self.data and self.inputs
        imgs = Variable(torch.cat((data['ref_img'][:,None], data['qry_imgs']), dim=1)).cuda()
        RTs = Variable(torch.cat((data['ref_RT'][:,None], data['qry_RTs']), dim=1)).cuda()
        Ks = Variable(torch.cat((data['ref_K'][:,None], data['qry_Ks']), dim=1)).cuda()
        self.actual_batch_size = imgs.shape[0]

        self.data = data
        self.inputs = {
            'imgs': imgs,
            'RTs': RTs,
            'Ks': Ks,
            'random_grid': False
        }

    def forward(self):
        # produces self.predicted
        lmk_pred_init, \
            lmk_cost_vol, global_grid, global_disp, global_Rot = self.model(**self.inputs)

        self.predicted = {
            'lmk_pred_init': lmk_pred_init,
            'lmk_cost_vol': lmk_cost_vol,
            'global_grid': global_grid,
            'global_disp': global_disp,
            'global_Rot': global_Rot,
        }

    # ---- compute losses / metrics ----

    def register_logger_meters(self):
        pass

    def compute_metrics(self):
        # produces loss during eval/test
        pass

    # ---- visualizations ----

    def register_visualizer(self):
        from utils.volumetric_lmk_visualizer import GlobalLandmarkVisualizer
        self.lmk_vis = GlobalLandmarkVisualizer(
            show=False, sph_r=self.args.vis_sphr,
            cmap_min=0.0, cmap_max=self.args.vis_cmap_max, cmap_name='jet',
            token_3d='points') # centimeters

    def save_visualizations(self, demo_id):
        # save all visualizations

        self.global_id = self.rel_i * self.batch_size + demo_id
        image_dir = self.image_output
        mesh_dir = self.mesh_output

        name = '%07d' % self.global_id # same only one instance
        self.demo_dir = join(mesh_dir, name)
        safe_mkdir(self.demo_dir)

        # metas
        self.save_metas(save_path=join(image_dir, f'input_{name}.json'), demo_id=demo_id)

        # images
        demo_imgs = self.save_imgs(save_path=join(image_dir, f'input_{name}.jpg'), demo_id=demo_id)

        # 2D landmark
        demo_visual_lmk = self.save_lmk_imgs(save_path=join(image_dir, f'lmk_{name}.jpg'), demo_id=demo_id, demo_imgs=demo_imgs)
        # example: visualize images in tensorboard
        # self.tf_logger.update_visuals(visuals={'lmk_2d': demo_visual_lmk.astype(np.float32)/255.}, step=i, mode='train')

        # 3D landmark and volume
        self.save_3d_predictions(demo_id=demo_id)

        # 3D separate visualization
        if self.global_id % 100 == 0:
            self.save_3d_debug(demo_id=demo_id)

    def save_metas(self, save_path, demo_id):
        """ save the meta info to json file, so that later we can track which input led to the results.
        """
        from utils.utils import save_json
        demo_metas = {
            'data_idx': self.data[ 'data_idx' ][demo_id].detach().cpu().numpy().tolist(),
            'ref_view_idx': self.data[ 'ref_view_idx' ][demo_id].detach().cpu().numpy().tolist(),
            'qry_view_idx': self.data[ 'qry_view_idx' ][demo_id].detach().cpu().numpy().tolist(),
            'ref_img_path': self.data[ 'ref_img_path' ][demo_id],
            'qry_img_paths': [ self.data[ 'qry_img_paths' ][bid][demo_id] for bid in range(len(self.data[ 'qry_img_paths' ])) ],
            'ref_mask_path': self.data[ 'ref_mask_path' ][demo_id],
            'qry_mask_paths': [ self.data[ 'qry_mask_paths' ][bid][demo_id] for bid in range(len(self.data[ 'qry_mask_paths' ])) ],
        }
        if 'algn_path' in self.data.keys():
            demo_metas['algn_path'] = self.data[ 'algn_path' ][demo_id]
        save_json(demo_metas, save_path)

    def save_imgs(self, save_path, demo_id):
        """ save the input images
        """
        demo_imgs = [ self.dataset_test.denormalize_image(
                        np.einsum( 'chw->hwc', self.inputs['imgs'][demo_id][idx].detach().cpu().numpy()) ) for idx in range(self.inputs['imgs'].shape[1]) ]
        imageio.imsave(save_path, (255. * np.concatenate(demo_imgs, axis=1) ).astype(np.uint8))
        return demo_imgs

    def save_lmk_imgs(self, save_path, demo_id, demo_imgs):
        """ save the visualization of projected predicted vertices on the input image
        """
        if args.dataset_id == 'coma_voca':
            raise NotImplementedError

        else:
            vert_pred = self.predicted['lmk_pred_init'][demo_id].detach().cpu().numpy()
            if "lmk_3d" in self.data.keys():
                vert_gt = self.data['lmk_3d'][demo_id].detach().cpu().numpy()
            else:
                vert_gt = None

            return self.lmk_vis.show_landmarks_2d_mv(
                lmk_pred=vert_pred, # note: hardcoded only visualize the lmk_pred_init
                lmk_gt=vert_gt,
                Ks=self.inputs['Ks'][demo_id].detach().cpu().numpy(),
                RTs=self.inputs['RTs'][demo_id].detach().cpu().numpy(),
                images=demo_imgs,
                save_path=save_path)

    def save_3d_predictions(self, demo_id):
        """ save the predicted vertices (referred as `lmk`) and optionally the ground truth vertices (if available)
        """
        # ground truth (dense mesh)
        if "algn_path" in self.data.keys() and self.data['algn_path'][demo_id] != '':
            from psbody.mesh import Mesh
            demo_algn = Mesh(filename=self.data['algn_path'][demo_id])
            # demo_algn = Mesh(v=data['algn_v'][demo_id].detach().cpu().numpy(), f=data['algn_f'][demo_id].detach().cpu().numpy())
            demo_algn.write_ply(join( self.demo_dir, 'algn.ply' ))

        # predicted sparse points
        vert_pred = self.predicted['lmk_pred_init'][demo_id].detach().cpu().numpy()
        save_npy(vert_pred, join(self.demo_dir, 'lmk_init.npy'))
        np.savetxt(join(self.demo_dir, 'lmk_init.txt'), vert_pred, delimiter=',')

        if "lmk_3d" in self.data.keys():
            self.lmk_vis.show_landmarks(
                lmk_pred=vert_pred,
                lmk_gt=self.data['lmk_3d'][demo_id].detach().cpu().numpy(),
                algn=None, show=False, save_path=join( self.demo_dir, 'lmk_init.ply' )) 
        else:
            from psbody.mesh import Mesh
            Mesh(v=vert_pred, f=[]).write_ply(join(self.demo_dir, 'lmk_init.ply'))

    def save_3d_debug(self, demo_id):
        """ these are whatever you want to visualize to dig deeper into the outputs
        here I visualized the predicted probability volumes
        """
        from utils.utils import save_binary_pickle
        # separate visualization
        for vis_idx in range(0, self.args.lmk_num, 5): # dimension index (landmark idx), skip 5
            # global
            self.lmk_vis.show_cost_volume(
                cost_vol=self.predicted['lmk_cost_vol'][demo_id][vis_idx].detach().cpu().numpy(),
                position_vol=self.predicted['global_grid'][demo_id][0].detach().cpu().numpy(),
                save_path=join( self.demo_dir, 'global_lmk_cost_vol_%02d.ply' % (vis_idx) ))

            # save volume data
            save_binary_pickle({
                    'cost_vol': self.predicted['lmk_cost_vol'][demo_id][vis_idx].detach().cpu().numpy(),
                    'position_vol': self.predicted['global_grid'][demo_id][0].detach().cpu().numpy(),
                }, join( self.demo_dir, 'global_lmk_cost_vol_%02d.pkl' % (vis_idx) ))

            # # local
            # self.lmk_vis.show_cost_volume(
            #     cost_vol=self.predicted['local_cost_vol'][demo_id][vis_idx].detach().cpu().numpy(),
            #     position_vol=self.predicted['local_grids'][-1][demo_id][vis_idx].detach().cpu().numpy(),
            #     save_path=join( self.demo_dir, 'local_lmk_cost_vol_%02d.ply' % (vis_idx) ))

        # combined visualization
        self.lmk_vis.show_cost_volumes(
            cost_vol=self.predicted['lmk_cost_vol'][demo_id].detach().cpu().numpy(),
            position_vol=self.predicted['global_grid'][demo_id][0].detach().cpu().numpy(),
            save_path=join( self.demo_dir, 'global_lmk_cost_vol_max.ply' ),
            summarize_method='max')

        save_binary_pickle({
            'cost_vol': self.predicted['lmk_cost_vol'][demo_id].detach().cpu().numpy(),
            'position_vol': self.predicted['global_grid'][demo_id][0].detach().cpu().numpy(),
            }, join( self.demo_dir, 'global_lmk_cost_vol_max.pkl' ))

        # self.lmk_vis.show_cost_volumes(
        #     cost_vol=self.predicted['local_cost_vol'][demo_id].detach().cpu().numpy(),
        #     position_vol=self.predicted['local_grids'][-1][demo_id].detach().cpu().numpy(),
        #     save_path=join( self.demo_dir, 'local_lmk_cost_vol_combined.ply' ),
        #     summarize_method='concate')

    def save_logs(self):
        pass

# -----------------------------------------------------------------------------

# arguments
parser = TestOptions()
args = parser.parse()

# set Tester
tester = Tester( args )
tester.initialize()
parser.save_json( join(tester.directory_output, 'options.json') )

# run
tester.run()
