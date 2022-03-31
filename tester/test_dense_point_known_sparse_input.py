'''
test the local (dense) stage of ToFu base
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
        # define specific parameters

        # data
        self.add_arg( cate='data', abbr='pd', name='processed-directory', type=str, default='') # "save_root"
        self.add_arg( cate='data', abbr='qvn', name='query-view-num', type=int, default=3) # the total view used is qvn + 1 (ref)
        self.add_arg( cate='data', abbr='ln', name='lmk-num', type=int, default=68)

        # model
        self.add_arg( cate='model', abbr='desc-dim', name='descriptor-dim', type=int, default=32)
        self.add_arg( cate='model', abbr='feat-arch', name='feature-arch', type=str, default='resnet')
        self.add_arg( cate='model', abbr='local-arch', name='local-arch', type=str, default='v2v')
        self.add_arg( cate='model', abbr='pretr-path', name='pretrained-path', type=str, default='')

        self.add_arg( cate='model', abbr='mesh-resamp-path', name='mesh-resample-path', type=str, default='')
        self.add_arg( cate='model', abbr='mesh-resamp-start', name='mesh-resample-start', type=int, default=3)
        self.add_arg( cate='model', abbr='mesh-resamp-end', name='mesh-resample-end', type=int, default=0)
        self.add_arg( cate='model', abbr='local-level', name='local-net-level', type=int, default=3)
        self.add_arg( cate='model', abbr='ge-type', name='global_embedding_type', type=str, default='none')
        self.add_arg( cate='model', abbr='pert', name='eps-perturb', type=float, default=0.1)
        self.add_arg( cate='model', abbr='pretr-refine-model-path', name='pretrained-refine-model-path', type=str, default='')

        self.add_arg( cate='model', abbr='lvd', name='local-voxel-dim', type=int, default=16)
        self.add_arg( cate='model', abbr='lvi', name='local-voxel-inc-list', type=list, default=[0.25, 0.125])
        self.add_arg( cate='model', abbr='no-normal', name='enforce-no-normal', type=bool, default=False)
        self.add_arg( cate='model', abbr='nsf', name='normal-scale-factor', type=float, default=1.0)
        self.add_arg( cate='model', abbr='ep', name='enable-profiling', type=bool, default=False)

        # visualization
        self.add_arg( cate='vis', abbr='sphr', name='vis-sphr', type=float, default=0.2)
        self.add_arg( cate='vis', abbr='cmap-max', name='vis-cmap-max', type=float, default=0.5)

        print( "added extra parameters" )

# -----------------------------------------------------------------------------

class Tester(BaseTester):

    def __init__(self, args):
        super().__init__(args)
        self.args = args

    # ---- model ----

    def register_model(self):
        import models.model_aligner.prototypes.model_dense_point_base as models
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
        from datasets.create_dataset_open import create_dataset
        self.dataset_test = create_dataset(self.args, mode='test', enforce_test_setting=True)
        self.dataloader_test = self.make_data_loader(self.dataset_test, cuda=True, shuffle=False)

    # ---- forward ----

    def feed_data(self, data):
        # receive a data instance from data loader to reorganize for model.forward format
        # records to self.data and self.inputs
        imgs = Variable(torch.cat((data['ref_img'][:,None], data['qry_imgs']), dim=1)).cuda()
        RTs = Variable(torch.cat((data['ref_RT'][:,None], data['qry_RTs']), dim=1)).cuda()
        Ks = Variable(torch.cat((data['ref_K'][:,None], data['qry_Ks']), dim=1)).cuda()
        self.actual_batch_size = imgs.shape[0]

        # get init vertices
        from utils.utils import load_npy
        if self.args.previous_case_dir:
            # load from precomputed data from the previous step
            prev_pts_path = join(self.args.previous_case_dir, 'obj', '%07d' % data['index'][0].item(), 'lmk_init.npy')
            prev_pts = load_npy(prev_pts_path)
            pts_init_v = Variable(torch.from_numpy(prev_pts)[None, :, :]).cuda()
        else:
            raise NotImplementedError

        self.data = data
        self.inputs = {
            'imgs': imgs,
            'RTs': RTs,
            'Ks': Ks,
            'pts_sparse': pts_init_v,
            'random_grid': False,
        }

    def forward(self):
        # produces self.predicted

        algn_v_pred_list = self.model(**self.inputs)

        # note: the network is currently only upsampling and refining the meshes to sub-highest level (3000 vertices)
        # as we observe diminishing effects of the learnt network. Therefore here we manually upsample (but do not refine) the mesh
        # to the highest 10000 vertices.
        # upsample 3000 pts to 10000 pts, without network refinement
        algn_v_pred_highest = self.model.module.densify_net.mr.upsample(
            algn_v_pred_list[-1],
            no_normals=self.args.enforce_no_normal  # TODO: shouldn't apply normal here if enforce not
        )
        algn_v_pred_list.append(algn_v_pred_highest)

        self.predicted = {
            'algn_v_pred_list': algn_v_pred_list,
        }

    # ---- compute losses / metrics ----

    def register_logger_meters(self):
        # register metric loggers
        pass

    def compute_metrics(self):
        # produces loss during eval/test
        pass

    # ---- visualizations ----

    def register_visualizer(self):
        from utils.volumetric_lmk_visualizer import GlobalLandmarkVisualizer
        from utils.alignment_visualizer import AlignmentVisualizer
        self.lmk_vis = GlobalLandmarkVisualizer(
            show=False, sph_r=self.args.vis_sphr,
            cmap_min=0.0, cmap_max=self.args.vis_cmap_max, cmap_name='jet',
            token_3d='points') # centimeters
        self.mesh_vis = AlignmentVisualizer(cmap_min=0.0, cmap_max=self.args.vis_cmap_max, cmap_name='jet', show=False)

        # # hack resize
        # if self.args.dataset_id == 'renderpeople':
        #     resize_factor = 2.0
        # if self.args.dataset_id == 'coma_voca_reorg':
        #     resize_factor = 2.0
        # else:
        #     resize_factor = 16.0

        # from utils.alignment_renderer_sr import AlignmentRendererSR
        # self.renderer = AlignmentRendererSR(cmap_min=0.0, cmap_max=self.args.vis_cmap_max, cmap_name='jet',
        #     resize_factor=resize_factor, dataset=self.args.dataset_id)

    def save_visualizations(self, demo_id):
        # save all visualizations

        self.global_id = self.rel_i * self.batch_size + demo_id
        image_dir = self.image_output
        mesh_dir = self.mesh_output
        render_dir = self.render_output

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
        self.save_3d_predictions(demo_id=demo_id, render_path=join(render_dir, f'{name}.jpg'))

        # # 3D separate visualization
        # if self.global_id % 100 == 0:
        #     self.save_3d_debug(demo_id=demo_id)

    def save_metas(self, save_path, demo_id):
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
            vert_pred = [vert[demo_id].detach().cpu().numpy() for vert in self.predicted['algn_v_pred_list']]
            if "algn_v_gt_list" in self.data.keys():
                vert_gt = [vert[demo_id].detach().cpu().numpy() for vert in self.data['algn_v_gt_list']]
            else:
                vert_gt = None

            return self.lmk_vis.show_landmarks_2d_mv(
                lmk_pred=vert_pred,
                lmk_gt=vert_gt,
                Ks=self.inputs['Ks'][demo_id].detach().cpu().numpy(),
                RTs=self.inputs['RTs'][demo_id].detach().cpu().numpy(),
                images=demo_imgs,
                save_path=save_path)

    def save_3d_predictions(self, demo_id, render_path):
        from psbody.mesh import Mesh
        from utils.utils import save_npy, save_binary_pickle

        level_num = len(self.predicted['algn_v_pred_list'])
        metric_names = ['s2m', 'm2s', 'v2v']

        # initialization (from previous global/sparse stage)
        if self.args.dataset_id == 'coma_voca':
            raise NotImplementedError
        else:
            demo_algn_init = Mesh(
                v=self.inputs['pts_sparse'][demo_id].detach().cpu().numpy(),
                f=self.model.module.densify_net.mr.faces[3].detach().cpu().numpy())  # (638, 3)

        demo_algn_init.write_ply(join( self.demo_dir, 'init.ply' ))
        save_npy(self.inputs['pts_sparse'][demo_id].detach().cpu().numpy(),
            join(self.demo_dir, f'init.npy'))

        # gt-scan
        from os.path import exists
        from psbody.mesh import Mesh
        try:
            scan_path = self.data['scan_path'][demo_id]
            scan = Mesh(filename=scan_path)
            if not exists(join(self.demo_dir, f'scan.ply')):
                scan.write_ply(join(self.demo_dir, f'scan.ply'))
        except:
            scan = None
            print(f"cannot find the scan mesh")

        renders = []

        # init points
        if 'algn_v_init_highest' in self.data.keys():
            # gt
            demo_algn_gt = Mesh(
                v=self.data['algn_v_gt_highest'][demo_id].detach().cpu().numpy(),
                f=self.data[ 'algn_f' ][demo_id].detach().cpu().numpy())

            # pred
            demo_algn_pred = Mesh(
                v=self.data['algn_v_init_highest'][demo_id].detach().cpu().numpy(),
                f=self.data[ 'algn_f' ][demo_id].detach().cpu().numpy())

            # save meshes
            gt_scan = demo_algn_gt if scan is None else scan # hack

            demo_meshes, demo_metrics = self.mesh_vis.visualize_compare_hardcoded(gt_scan=gt_scan, gt_mesh=demo_algn_gt, mesh=demo_algn_pred)
            # demo_meshes[0].write_ply(join(self.demo_dir, f'level_init_gt_scan.ply'))
            demo_meshes[1].write_ply(join(self.demo_dir, f'level_init_gt_mesh.ply'))
            demo_meshes[2].write_ply(join(self.demo_dir, f'level_init_algn_pred.ply'))
            # demo_meshes[3].write_ply(join(self.demo_dir, f'level_init_overlay.ply'))
            for mid in range(0, len(metric_names)):
                demo_meshes[4+mid].write_ply(join(self.demo_dir, f'level_init_{metric_names[mid]}.ply'))

            # save rendering
            # this_render = self.renderer.render_compare_pre_meshes(meshes=demo_meshes)
            # renders.append(this_render)

            # save metrics
            save_binary_pickle(
                data={'metric_names': metric_names, 'metrics': demo_metrics},
                filepath=join(self.demo_dir, f'level_init_metrics.npy'))

        # render results
        for idx in range(level_num):

            level = level_num - 1 - idx # level=0 if highest
            mesh_faces = self.model.module.densify_net.mr.faces[level].detach().cpu().numpy()

            # pred
            vert_pred = self.predicted['algn_v_pred_list'][idx][demo_id].detach().cpu().numpy()
            demo_algn_pred = Mesh(v=vert_pred, f=mesh_faces)

            # gt
            if "algn_v_gt_list" in self.data.keys():
                if self.args.dataset_id == 'coma_voca':
                    raise NotImplementedError
                else:
                    vert_gt = self.data['algn_v_gt_list'][idx][demo_id].detach().cpu().numpy()
                    demo_algn_gt = Mesh(v=vert_gt, f=mesh_faces)
            else:
                vert_gt = None
                demo_algn_gt = None

            # save values
            save_npy(vert_pred, join(self.demo_dir, f'level_{level}_algn_pred.npy'))

            # special
            if idx < level_num-1 and 'special' in self.inputs.keys():
                demo_algn_pred_pre_network = Mesh(
                    v=self.predicted['algn_v_pre_network_list'][idx][demo_id].detach().cpu().numpy(),
                    f=mesh_faces)
                demo_algn_pred_pre_network.write_ply(
                    join(self.demo_dir, f'level_{level}_algn_pred_pre_network.ply'))

            # TODO: render result mesh in checkboard texture

            # save meshes with colormapped visualizations
            # requires to have at least one type of gt mesh (mesh/scan)
            if demo_algn_gt is not None:
                gt_scan = demo_algn_gt if scan is None else scan # hack
                demo_meshes, demo_metrics = self.mesh_vis.visualize_compare_hardcoded(gt_scan=gt_scan, gt_mesh=demo_algn_gt, mesh=demo_algn_pred)
                # demo_meshes[0].write_ply(join(self.demo_dir, f'level_{level}_gt_scan.ply'))
                demo_meshes[1].write_ply(join(self.demo_dir, f'level_{level}_gt_mesh.ply'))
                demo_meshes[2].write_ply(join(self.demo_dir, f'level_{level}_algn_pred.ply'))
                # demo_meshes[3].write_ply(join(self.demo_dir, f'level_{level}_overlay.ply'))
                for mid in range(0, len(metric_names)):
                    demo_meshes[4+mid].write_ply(join(self.demo_dir, f'level_{level}_{metric_names[mid]}.ply'))

                # # save rendering
                # this_render = self.renderer.render_compare_pre_meshes(meshes=demo_meshes)
                # renders.append(this_render)

                # save metrics
                save_binary_pickle(
                    data={'metric_names': metric_names, 'metrics': demo_metrics},
                    filepath=join(self.demo_dir, f'level_{level}_metrics.npy'))

                # # error visualization
                # self.lmk_vis.show_landmarks(
                #     lmk_pred=self.predicted['algn_v_pred_list'][idx][demo_id].detach().cpu().numpy(),
                #     lmk_gt=self.data['algn_v_gt_list'][idx][demo_id].detach().cpu().numpy(),
                #     algn=None, show=False, save_path=join( self.demo_dir, f'level_{level}_pts_pred.ply' ))

            else:
                # simply save the predicted meshes
                demo_algn_pred.write_ply(join(self.demo_dir, f'level_{level}_algn_pred.ply'))

        # examples

        #     # save metrics
        #     save_binary_pickle(
        #         data={'metric_names': metric_names, 'metrics': demo_metrics},
        #         filepath=join(self.demo_dir, f'level_highest_metrics.npy'))

        #     # save render
        #     import imageio
        #     # imageio.imwrite(render_path, (255. * np.concatenate((renders), axis=0)).astype(np.uint8))


        # else:
        #     # pred
        #     demo_algn_pred = Mesh(
        #         v=self.predicted['algn_v_pred_highest'][demo_id].detach().cpu().numpy(),
        #         f=self.model.module.densify_net.mr.faces[0])
        #         # f=self.data[ 'algn_f' ][demo_id].detach().cpu().numpy())
        #     demo_algn_pred.write_ply(join(self.demo_dir, f'level_highest_gt_mesh.ply'))

    def save_3d_debug(self, demo_id):
        # separate visualization for debugging purposes
        raise NotImplementedError

    def save_logs(self):
        # save (summaries of) metrics
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
