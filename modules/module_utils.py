"""
utils for modules / models

tianye li
Please see LICENSE for the licensing information
"""
import torch
import torch.nn as nn
from torch.nn import init

# -----------------------------------------------------------------------------

def init_weights(net, init_type='normal', gain=0.02, verbose=False):
    """initialize parameters by certain distribution
    Args:
        init_method: allowed: 'normal', 'xavier', 'kaiming', 'orthogonal', 'nothing'
        gain: float
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'nothing':
                pass # do nothing
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    if verbose: print('initialize network with %s' % init_type)
    net.apply(init_func)

# -----------------------------------------------------------------------------

def init_net(net, init_type='normal', gpu_ids=[], pretrained_model_path='', option='' ):

    # regular initialization
    init_weights(net, init_type)

    # load pretrained
    if pretrained_model_path:
        if option == 'load_from_resnet':
            net.initialize_from_pretrained_model( pretrained_model_path, load_from_resnet=True ) # assume this 'net' has this member function
        else:
            net.initialize_from_pretrained_model( pretrained_model_path )

    # set to gpus
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    return net

# -----------------------------------------------------------------------------

def copy_weights(src_path, dst_net, keywords=None, name_maps=[lambda x: x], verbose=False):
    """ copy weights from source path to dst model
    Args:
        src_path: path to source model
        dst_net: destination network (instance of nn.Module)
        keyword: list of string (as prefixes of keys)
        name_maps: list of str mapping from destination name to source name
            e.g. src_name = 'sparse_point_net.local_net.*'
                 dst_name = 'densify_net.local_net.*'
            then the map should be: lambda dst: 'sparse_point_net.' + dst.replace('densify_net.', '')
    """
    if isinstance(src_path, str):
        src_dict = torch.load(src_path)['model']
    else:
        src_dict = src_path # assume this is already loaded model dict
    dst_dict = dict(dst_net.named_parameters())

    if keywords is None: keywords = [''] # all strings contains ''
    if isinstance(keywords, str): keywords = [keywords]

    successful_dst_names = []

    for ki, keyword in enumerate(keywords):
        for mi, mp in enumerate(name_maps):
            for dst_name in dst_dict.keys(): # dst
                src_name = mp(dst_name)
                if src_name in src_dict.keys() and keyword in src_name:
                    if dst_dict[ dst_name ].data.shape == src_dict[ src_name ].data.shape:
                        dst_dict[ dst_name ].data.copy_( src_dict[ src_name ].data )
                        if verbose: print( "layer '%s' parms copied" % ( dst_name ) )
                        successful_dst_names.append(dst_name)
                    else:
                        if verbose: print( "layer '%s' parms shape doesn't match, not copied" % ( dst_name ) )

    return successful_dst_names

# -----------------------------------------------------------------------------

def hack_remove_sparse_matrices(model_path, save_path=None):
    # a hack: remove the sparse matrices in saved model
    # pytorch currently (v1.5) has issue in loading the saved model which contains sparse matrices
    # input path is a pth.tar file

    src_dict = torch.load(model_path)['model']
    print(f'loaded {model_path}')

    try:
        del src_dict['densify_net.mr.D_0']
        del src_dict['densify_net.mr.U_0']
        del src_dict['densify_net.mr.A_0']
        del src_dict['densify_net.mr.D_1']
        del src_dict['densify_net.mr.U_1']
        del src_dict['densify_net.mr.A_1']
        del src_dict['densify_net.mr.D_2']
        del src_dict['densify_net.mr.U_2']
        del src_dict['densify_net.mr.A_2']
    except:
        raise RuntimeError(f"error in deleting the densify_net.mr.* sparse matrices; probably those tensors do not exist in current model.")

    if save_path:
        torch.save({'model': src_dict}, save_path)
    else:
        from os.path import basename, join
        bname = basename(model_path)
        base_dir = '/'.join(model_path.split('/')[0:-1])
        save_path = join(base_dir, 'safe_'+bname)
        torch.save({'model': src_dict}, save_path)
    print(f'saved processed model at {save_path}')