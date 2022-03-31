'''
base tester
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
from utils.utils import AverageMeter

# -----------------------------------------------------------------------------

class BaseTester():

    def __init__(self, args):
        self.args = args

    def initialize(self):
        self.control_seeds()
        self.check_requirements()
        self.register_model()
        # self.load_model()
        self.mkdirs() # change: mkdir according to model name (iter, ...)
        self.register_dataset()
        self.register_logger()
        self.register_visualizer()

    # ----------------------
    # meta

    def control_seeds(self):
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)

    def mkdirs(self):

        if not hasattr(self, 'model') or not hasattr(self, 'model_name'):
            raise RuntimeError(f"must have created and loaded the model before mkdirs()")

        if not hasattr(self, 'experiment_name'):
            if self.args.experiment_id:
                experiment_name = self.args.experiment_id
            else:
                experiment_name = self.args.model_path.split('/')[-2]
            # required: put model name as suffix to experiment_name
            experiment_name = experiment_name + "_" + self.model_name
        else:
            experiment_name = self.experiment_name
        print(f'adopted experiment_name = {experiment_name}')

        if self.args.dataset_id:
            dataset_name = self.args.dataset_id
        else:
            dataset_name = self.args.dataset_directory.split('/')[-1]

        output_directory = self.args.model_directory # model training paths - won't be used
        test_directory = self.args.test_directory # test results paths

        self.directory_output = join( test_directory, experiment_name, dataset_name )
        os.makedirs(test_directory, exist_ok=True)
        os.makedirs(join( test_directory, experiment_name), exist_ok=True)
        os.makedirs(self.directory_output, exist_ok=True)
        self.image_output = join(self.directory_output, 'pic')
        os.makedirs(self.image_output, exist_ok=True)
        self.mesh_output = join(self.directory_output, 'obj')
        os.makedirs(self.mesh_output, exist_ok=True)
        self.render_output = join(self.directory_output, 'render')
        os.makedirs(self.render_output, exist_ok=True)

        print( "created testing output folder for:" )
        print( " - experiment: %s" % ( experiment_name ) )
        print( " - dataset: %s" % ( dataset_name ) )
        print( " - output root: %s" % ( test_directory ) )
        print( " - output diretory: %s" % ( self.directory_output ) )

    # ----------------------
    # special requirements

    def check_requirements(self):

        # enforce requirements
        if self.args.batch_size != 1:
            print(f"test batch_size must be 1 provided {self.args.batch_size}, enforcing to 1 now...")
            self.args.batch_size = 1

        # enforce dataloader not using too many threads
        self.args.thread_num = 2
        print(f"enforcing thread_num to 2 now...")

    # ----------------------
    # model

    def register_model(self):
        self.model = None # instance of torch.nn.DataParallel(model).cuda()

    def set_train(self):
        self.model.train(True)
        print(f"set model to mode 'train'")

    def set_eval(self):
        self.model.train(False)
        print(f"set model to mode 'eval'")

    def set_test(self):
        self.set_eval()

    def load_model(self, enforce_traditional_load=False, verbose=False):
        if self.model is None: raise RuntimeError( 'model is not yet specified.' )

        print( "loading model..." )
        tic = time.time()

        # find the latest model
        from os.path import basename
        model_bname = basename(self.args.model_path)

        if '.pth.tar' in model_bname:
            model_path = self.args.model_path
            self.model_name = model_bname.replace('.pth.tar', '')
        elif 'latest' == model_bname:
            from utils.utils import get_latest_model
            bname = basename(self.args.model_path)
            model_dir = self.args.model_path.replace(bname, '')
            model_path = get_latest_model(model_dir, verbose=True)
            self.model_name = basename(model_path).replace('.pth.tar', '')
        else:
            raise RuntimeError(f"invalid model path with basename = {model_bname}")

        # load
        try:
            state_dicts = torch.load(model_path)
            if state_dicts.get('model', None) is not None:
                self.model.load_state_dict(state_dicts['model'])
            else:
                self.model.load_state_dict(state_dicts)
            print( "loaded model from %s (%.1f sec)" % ( model_path, time.time()-tic ) )

        except:
            if enforce_traditional_load:
                raise RuntimeError(f"error in loading model from {model_path}")

            else:
                # issue: sparse tensor parameter somehow cannot be copied
                # see: https://github.com/pytorch/pytorch/issues/22616
                # current work-around:
                from modules.module_utils import copy_weights
                if model_path not in ['', None]:
                    copy_weights(src_path=model_path, dst_net=self.model,
                        keywords=['feature_net', 'local_net', 'global_net'],  # hardcoded
                        name_maps=[
                            lambda dst: dst,
                            # lambda dst: 'sparse_point_net.' + dst.replace('densify_net.', '')
                        ], verbose=verbose)
                    print( "(SPECIAL) loaded model from %s (%.1f sec)" % ( model_path, time.time()-tic ) )

        # by default: set test mode
        self.set_eval()

    # ----------------------
    # dataset

    def worker_init_fn(self, worker_id):
        # to properly randomize:
        # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def make_data_loader(self, dataset, cuda=True, shuffle=True):

        self.args.thread_num = 1
        print(f"DEBUG: self.args.thread_num = {self.args.thread_num}")

        kwargs = {'num_workers': self.args.thread_num, 'pin_memory': True} if cuda else {}
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle,
                                            worker_init_fn=self.worker_init_fn, **kwargs)

    def register_dataset(self):
        self.dataset_test = None
        self.dataloader_test = None

    # ----------------------
    # logger

    def register_logger(self):
        # text loggers
        from utils.simple_logger import Logger
        self.logger = Logger( os.path.join(self.directory_output, 'losses_test.txt') )
        # # tensorboard logger
        # from utils.tensorboard_logger import Logger as TensorboardLogger        
        # self.tf_logger = TensorboardLogger(os.path.join(self.directory_output, 'logs'))

    def register_loss_meters(self):
        pass

    def save_logs(self):
        # save logger meters informations
        pass

        # # example
        # from utils.utils import save_json, load_npy
        # metrics = self.logger.summarize_meters(method='avg')
        # save_json(metrics, filepath=join(self.directory_output, 'metric_summary.json'))
        # records = self.logger.summarize_records()
        # save_json(records, filepath=join(self.directory_output, 'metric_records.json'))

    # ----------------------
    # visualizer

    def register_visualizer(self):
        self.visualizer = None

    # ----------------------
    # main components

    def feed_data(self, data, mode="test"):
        # consumes a data instance from data loader to reorganize for model.forward format
        # produces self.data and self.inputs as dict
        raise NotImplementedError(f"feed_data() not defined yet")

        # example
        self.data = data
        self.actual_batch_size = data['imgs'].shape[0]
        self.inputs = {
            # reorganize data here, should fit the model.forward() input args
        }

    def forward(self):
        # consumes self.inputs
        # produces self.predicted
        raise NotImplementedError(f"forward() not defined yet")

        # example
        stuff = self.model(**self.inputs)
        self.predicted = {
            # save your stuff here
        }

    def compute_metrics(self):
        # consumes self.inputs, self.data, self.predicted
        # produces and produces metrics during validation
        raise NotImplementedError(f"compute_metrics() not defined yet")

        # # example
        # loss_1 = self.args.lambda_loss_1 * loss_func_1(pred=self.predicted['stuff'], gt=self.data['stuff'], norm_type=self.args.type_stuff)
        # # ...

        # # losses
        # loss = loss_1 # + ...

        # # record
        # self.logger.set_meter('loss_1', loss_1.data.item(), self.batch_size)
        # # ...
        # self.logger.set_meter('total', loss.data.item(), self.batch_size)

    def save_visualizations(self, demo_id, mode='test'):
        # produces and saves visualization
        # demo_id: idx within the batched data
        raise NotImplementedError(f"save_visualizations() not defined yet")

        if mode not in ['test']:
            raise RuntimeError(f"invalid mode = {mode}")
        image_dir = getattr(self, f'image_output_{mode}')
        mesh_dir = getattr(self, f'mesh_output_{mode}')
        # ...

    # ----------------------
    # main processes

    def test(self):
        test_tic = time.time()
        self.logger.clear_meters()

        self.model.train(False)
        for rel_i, data in enumerate( self.dataloader_test ):
            self.rel_i = rel_i

            # set test data range
            if self.batch_size == 1 and self.args.enforce_test_data_range != []:
                range_min = self.args.enforce_test_data_range[0]
                range_max = self.args.enforce_test_data_range[1]
                if not rel_i in range(range_min, range_max):
                    print(f"skip batch {rel_i}, required [{range_min}, {range_max}]")
                    continue

            # data
            self.feed_data(data) # data --> self.data, self.inputs, self.actual_batch_size
            if self.inputs is None:
                print(f"BaseTester::test(): batch {rel_i} invalid, skipping")
                continue

            # forward
            with torch.no_grad():
                self.forward()
                self.compute_metrics()

            # visualize
            for demo_id in range(self.actual_batch_size):
                self.save_visualizations(demo_id)

            print(f"test done {self.global_id}/{len(self.dataset_test)}")

        # log
        self.save_logs()

    def run(self):
        self.end = time.time()
        self.batch_time = AverageMeter()
        self.batch_time_data = AverageMeter()

        # loss meters
        self.register_logger_meters()
        self.batch_size = self.args.batch_size

        # test
        self.test()
