'''
base model
for model aligner

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
import time

# -----------------------------------------------------------------------------

class BaseModel(nn.Module):

    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()

        # ---- properties ----
        # self.member = member
        # self.architecture = architecture

        # ---- modules ----
        # by default, use "model" as the field name. you could of course customize.
        self.module_names = ['model']
        self.model = None # should be an instance of nn.Module

    def initialize(self, init_method='kaiming', model_path=None, verbose=False):
        """initialize parameters by certain distribution
        Args:
            init_method: allowed: 'normal', 'xavier', 'kaiming', 'orthogonal', 'nothing'
            model_path: path to pretrained model
            verbose: if print out info
        """
        from modules.module_utils import init_weights
        for name in self.module_names:
            init_weights(getattr(self, name), init_type=init_method, verbose=verbose)

        if model_path is not None and isinstance(model_path, str):
            self.load_special(model_path, verbose=verbose)

    def load_special(self, model_path, verbose=False):
        """load model from pretrained model of not exactly this class
        i.e. you would need to copy some pretrained weights to this class
        - you may also decide specific method given your setting, e.g. self.architecture
        - always called if model_path is string
        """
        # # copy
        # src_parms_dict = state_dicts['model']
        # dst_parms_dict = dict( self.named_parameters() )
        # for dst_name in dst_parms_dict.keys(): # dst
        #     # dst_name = e.g.        'feature_net.resnet34_8s.conv1.weight'
        #     # src_name = e.g. 'module.feature_net.resnet34_8s.conv1.weight'
        #     src_name = 'module.' + dst_name
        #     if src_name in src_parms_dict.keys():
        #         dst_parms_dict[ dst_name ].data.copy_( src_parms_dict[ src_name ].data )
        #         if verbose: print( "layer '%s' parms copied" % ( dst_name ) )
        pass

    def load(self, model_path, verbose=False):
        """load model from pretrained model of exactly this class (often called manually)
        """
        if verbose: print( "initializing from pretrained model..." )
        tic = time.time()

        try:
            state_dicts = torch.load(model_path)
            self.load_state_dict(state_dicts['model'], strict=False)
            if verbose: print( "initialized model from pretrained %s (%.1f sec)" % ( model_path, time.time()-tic ) )

        except:
            from modules.module_utils import copy_weights
            if model_path not in ['', None]:
                copy_weights(src_path=model_path, dst_net=self,
                    keywords=None,
                    name_maps=[
                        lambda dst: dst,
                        # lambda dst: 'sparse_point_net.' + dst.replace('densify_net.', '')
                    ], verbose=True)
                if verbose: print( "(SPECIAL) initialized model from pretrained %s (%.1f sec)" % ( model_path, time.time()-tic ) )

    def save(self, model_dir, iter_num):
        raise NotImplementedError
        # for module_name in self.module_names:
        #     model_path = os.path.join(model_dir, '%s_%07d.pth.tar' % (module_name, iter_num) )
        #     torch.save({
        #         'model': getattr(self, module_name).state_dict(),
        #         'optimizer_model': trainer.optimizer_model.state_dict(),
        #         }, model_path)

    def parms(self):
        parms_list = []
        for name in self.module_names:
            parms_list += list(getattr(self, name).parameters())
        return parms_list

    def optimizable_parms(self):
        """ parameters to be optimized. Default: all parameters
        This function can be override by child-classes
        """
        return self.parms()

    def named_parms(self):
        parms_dict = {}
        for name in self.module_names:
            parms_dict[name] = dict(getattr(self, name).named_parameters())
        return parms_dict

    def print(self, verbose=2):
        from utils.debug import print_network
        for name in self.module_names:
            print_network(getattr(self, name), verbose=verbose)

    def print_setting(self):
        # print out information as in **kwargs
        pass

    def forward(self, x):
        raise NotImplementedError(f"forward function not yet specified")
