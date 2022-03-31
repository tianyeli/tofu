'''
util

tianye li
Please see LICENSE for the licensing information
'''
from __future__ import print_function
import numpy as np
import sys
import os
from os.path import exists, join
from PIL import Image
import torch
from collections import OrderedDict

# -----------------------------------------------------------------------------

def load_binary_pickle( filepath ):
    if sys.version_info[0] < 3:
        import cPickle as pickle
    else:
        import pickle

    with open( filepath, 'rb' ) as f:
        data = pickle.load( f )
    return data

# -----------------------------------------------------------------------------

def save_binary_pickle( data, filepath ):
    if sys.version_info[0] < 3:
        import cPickle as pickle
    else:
        import pickle
    with open( filepath, 'wb' ) as f:
        pickle.dump( data, f )

# -----------------------------------------------------------------------------

def save_npy( data, filepath ):
    with open( filepath, 'wb' ) as fp:
        np.save( fp, data ) 

# -----------------------------------------------------------------------------

def load_npy( filepath ):
    data = None
    with open( filepath, 'rb' ) as fp:
        data = np.load( fp ) 
    return data

# -----------------------------------------------------------------------------

def load_json( filepath ):
    import json
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data

# -----------------------------------------------------------------------------

def save_json(data, filepath, indent=4, verbose=False):
    import json
    with open(filepath, 'w') as fp:
        json.dump(data, fp, indent=indent)
    if verbose: print(f"saved json file at: {filepath}")

# -----------------------------------------------------------------------------

def get_extension( file_path ):
    import os.path
    return os.path.splitext( file_path )[1] # returns e.g. '.png'

# -----------------------------------------------------------------------------

def safe_mkdir( file_dir, enable_777=False, recursive=True ):
    if sys.version_info[0] < 3:
        if not os.path.exists( file_dir ):
            os.mkdir( file_dir )
    else:
        from pathlib import Path
        path = Path(file_dir)
        path.mkdir(parents=recursive, exist_ok=True)
    if enable_777:
        chmod_777( file_dir )

# -----------------------------------------------------------------------------

def value2color( data, vmin=0, vmax=0.001, cmap_name='jet' ):
    # 'data' is np.array in size (H,W)
    import matplotlib as mpl
    import matplotlib.cm as cm

    norm = mpl.colors.Normalize( vmin=vmin, vmax=vmax )
    cmap = cm.get_cmap( name=cmap_name )
    colormapper = cm.ScalarMappable( norm=norm, cmap=cmap )
    rgba = colormapper.to_rgba( data.astype(np.float) )
    color_3d = rgba[...,0:3]
    return color_3d

# -----------------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        # assume val is the average value for a batch
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# -----------------------------------------------------------------------------

class AdvancedMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    @property
    def avg(self):
        return np.sum(np.asarray(self.vals_weighted), axis=0) / np.asarray(self.bs).sum()

    @property
    def median(self):
        return np.median(np.asarray(self.vals_weighted), axis=0) # only works if bs = 1 for all entries

    @property
    def max(self):
        return np.max(np.asarray(self.vals_weighted), axis=0) # only works if bs = 1 for all entries

    @property
    def min(self):
        return np.min(np.asarray(self.vals_weighted), axis=0) # only works if bs = 1 for all entries

    @property
    def percentile25(self):
        return np.percentile(np.asarray(self.vals_weighted), 25, axis=0) # only works if bs = 1 for all entries

    @property
    def percentile75(self):
        return np.percentile(np.asarray(self.vals_weighted), 75, axis=0) # only works if bs = 1 for all entries

    @property
    def val(self):
        return self.vals[-1] # last value

    @property
    def sum(self):
        return np.sum(np.asarray(self.vals_weighted), axis=0)

    @property
    def count(self):
        return np.asarray(self.bs).sum()

    @property
    def records(self):
        return {
            'vals': self.vals.tolist() if isinstance(self.vals, np.ndarray) else self.vals,
            'vals_weighted': self.vals_weighted.tolist() if isinstance(self.vals_weighted, np.ndarray) else self.vals_weighted,
            'bs': self.bs.tolist() if isinstance(self.bs, np.ndarray) else self.bs
        }

    def reset(self):
        self.vals = []
        self.vals_weighted = []
        self.bs = []

    def update(self, val, n=1):
        # # assume val is the average value for a batch
        if isinstance(val, list):
            val_w = [ el * n for el in val ]
        elif isinstance(val, np.ndarray):
            val_w = ( val * n ).tolist()
            val = val.tolist()
        else:
            val_w = [ val * n ]
            val = [ val ]

        self.vals.append(val)
        self.vals_weighted.append(val_w)
        self.bs.append(n)