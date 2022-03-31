'''
feature 2d network
extract feature from 2d images
tianye
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_aligner.base_model import BaseModel

# -----------------------------------------------------------------------------

class Model(BaseModel):

    def __init__(self, input_ch, output_ch, architecture, **kwargs):
        super(Model, self).__init__()

        # ---- properties ----
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.architecture = architecture

        # ---- modules ----
        self.module_names = ['model']

        # feature extractor for 2d
        if self.architecture == 'uresnet':
            import modules.resnet_dilated as resnet_dilated
            self.model = resnet_dilated.Resnet34_8s_skip(num_classes=self.output_ch, pretrained=True)
        else:
            raise RuntimeError( "unrecognizable architecture: %s" % ( self.architecture ) )

    def load_special(self, model_path, verbose=False):
        if self.architecture == 'uresnet':
            # you may load the model accordingly for each architecture
            pass
        else:
            raise RuntimeError( "unrecognizable architecture: %s" % ( self.architecture ) )

    def print_setting(self):
        print("-"*40)
        print(f"name: feature_net_2d")
        print(f"\t- input_ch: {self.input_ch}")
        print(f"\t- output_ch: {self.output_ch}")
        print(f"\t- architecture: {self.architecture}")

    def forward(self, x):
        '''compute 2d feature maps given images
        Args:
            x: tensor in (B, C, H', W'). H', W' are orig size
        Returns:
            x: tensor in (B, F, H, W), note the height and width might change
        '''
        # meta
        bs, ic, ih, iw = x.shape
        device = x.device
        assert ic == self.input_ch, f"unmatched input image channel {ic}, expected {self.input_ch}"

        # run
        x = self.model(x)
        return x

# -----------------------------------------------------------------------------

def test():

    device = torch.device('cuda')
    args = {
        'input_ch': 3,
        'output_ch': 8,
        'architecture': 'hrnet18',
    }
    model = Model(**args)
    model = model.to( device )
    model.print()
    model.initialize(init_method='kaiming', verbose=True)

    # forward
    B = 1
    H, W = 224, 224
    x = torch.randn(B, args['input_ch'], H, W).to(device)
    feat = model.forward(x)

    import ipdb; ipdb.set_trace()

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    test()