'''
base test options
based on options.*
tianye li
Please see LICENSE for the licensing information'''
import argparse
import os
import torch

# -----------------------------------------------------------------------------

class BaseTestOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  
        self.hierachy = {}

    def parse(self):
        self.initialize()
        _, _ = self.parser.parse_known_args()
        opt = self.parser.parse_args()
        self.opt = opt

        # possibly override if option_path is provided
        # this will override all args provided in python command
        if getattr(opt, 'option_path') != '':
            self.load_from_json( opt.option_path )

        self.print_options(opt)
        return self.opt

    @staticmethod
    def cmd2name(cmd):
        return cmd.replace('-', '_')

    @staticmethod
    def name2cmd(name):
        return name.replace('_', '-')

    def add_arg(self, cate, abbr, name, type, default):
        self.parser.add_argument('-'+abbr, '--'+self.name2cmd(name), type=type, default=default)
        if cate not in self.hierachy.keys():
            self.hierachy[cate] = []
        self.hierachy[cate].append( name )

    def initialize(self):

        # base
        self.add_arg( cate='base', abbr='md',  name='model-directory', type=str, default='' ) # equivalent to output_directory
        self.add_arg( cate='base', abbr='eid', name='experiment-id', type=str, default='' )
        self.add_arg( cate='base', abbr='s',   name='seed', type=str, default=0 )
        self.add_arg( cate='base', abbr='g',   name='gpu', type=str, default=0 )
        self.add_arg( cate='base', abbr='op',  name='option-path', type=str, default='' )

        # test
        self.add_arg( cate='test', abbr='td', name='test-directory', type=str, default='' ) # test results will be saved here
        self.add_arg( cate='test', abbr='mp', name='model-path', type=str, default='')
        self.add_arg( cate='test', abbr='test-size', name='test-data-size', type=int, default=0) # define test data size
        self.add_arg( cate='test', abbr='order-path', name='load-order-path', type=str, default='')
        self.add_arg( cate='test', abbr='prev-dir', name='previous-case-dir', type=str, default='') # dir to previous step test results
        # e.g. /home/tli/Dropbox/DenseFaceTracking/tests/20200503_sparse_baseline_8vw_v2v_init_only_beta_3e-3_lmk341/LightStageFaceDB
        self.add_arg( cate='test', abbr='etdr', name='enforce-test-data-range', type=list, default=[]) # define test data size

        # train
        self.add_arg( cate='train', abbr='b',   name='batch-size', type=int, default=4)
        self.add_arg( cate='train', abbr='pf',  name='print-freq', type=int, default=100)
        self.add_arg( cate='train', abbr='df',  name='demo-freq', type=int, default=100)

        # data
        self.add_arg( cate='data', abbr='di', name='dataset-id', type=str, default='')
        self.add_arg( cate='data', abbr='dd', name='dataset-directory', type=str, default='')
        self.add_arg( cate='data', abbr='sd', name='split-directory', type=str, default='')
        self.add_arg( cate='data', abbr='ds', name='dataset-size', type=int, default=0)
        self.add_arg( cate='data', abbr='thread', name='thread-num', type=str, default=4)

        # model
        # ...

        # loss
        # self.add_arg( cate='loss', abbr='lks', name='lambda-keypoint-silh', type=float, default=0.)

        # register extra options
        self.initialize_extra()

        self.initialized = True

    def initialize_extra(self):
        # to be defined
        pass

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        categories = self.hierachy.keys()
        for cate in categories:
            message += '\n[{:}]:\n'.format(cate)
            for k in self.hierachy[cate]:
                v = getattr( opt, self.cmd2name(k) )
                comment = ''
                default = self.parser.get_default(self.cmd2name(k))
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        self.message = message

    def save_json(self, save_path):
        data = {}
        for cate in self.hierachy.keys():
            data[cate] = {}
            for k in self.hierachy[cate]:
                data[cate][self.cmd2name(k)] = getattr( self.opt, self.cmd2name(k) )
        import json
        with open(save_path, 'w') as fp:
            json.dump(data, fp, indent=4)
        print( "saved options to json file: %s" % (save_path) )

    def load_from_json(self, json_path):
        import json
        with open(json_path) as fp:
            data = json.load(fp)
        for cate in data.keys():
            for k in data[cate].keys():
                setattr( self.opt, self.cmd2name(k), data[cate][self.cmd2name(k)] )
        print( "options overridden by json file: %s" % (json_path) )