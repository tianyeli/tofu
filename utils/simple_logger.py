'''
simple logger

tianye li
Please see LICENSE for the licensing information
'''
import time
import os
from collections import OrderedDict
from utils.utils import AverageMeter, AdvancedMeter

# -----------------------------------------------------------------------------

class Logger(object):
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Loss (%s) ================\n' % now)

        # cache
        # todo: added timer and loss meters into Logger class
        self.meters = OrderedDict()
        self.batch_time = AverageMeter() # not used for now
        self.batch_time_data = AverageMeter() # not used for now

        # self.losses_dict = OrderedDict() # not used for now

    # ------ timers ------

    def tic():
        self.tic = time.time()

    def batch_tic():
        self.batch_tic = time.time()

    # ------ meters ------

    def add_meter(self, name):
        # self.meters[ name ] = AverageMeter()
        self.meters[ name ] = AdvancedMeter()

    def set_meter(self, name, value, batch_size=1):
        # value must be a str
        self.meters[ name ].update( value, batch_size )

    def clear_meter(self, name):
        self.meters[ name ].reset()

    def add_meters(self, name_list):
        for name in name_list:
            self.add_meter(name)

    def summarize_meters(self, method='avg', debug=False):
        import numpy as np
        losses = OrderedDict()
        # mtds = method.split('_') # 'avg_median_min_max' -> ['avg', 'median', 'min', 'max']

        for name in self.meters.keys():
            this_loss = getattr( self.meters[ name ], method )
            if isinstance(this_loss, np.ndarray):
                this_ret = { kk: this_loss.tolist()[kk] for kk in range(this_loss.shape[0]) } if this_loss.shape[0] > 1 else this_loss[0] # special hack for np.array
            elif isinstance(this_loss, list):
                this_ret = { kk: this_loss[kk] for kk in range(len(this_loss)) } if len(this_loss) > 1 else this_loss[0] # special hack for np.array
            else:
                this_ret = this_loss

            losses[ name ] = this_ret
            if debug:
                print( f"summarized loss '{name}' = {losses[name]}" )
        return losses

    def summarize_records(self):
        import numpy as np
        records = OrderedDict()
        for name in self.meters.keys():
            this_records = getattr( self.meters[name], 'records' )
            records[ name ] = this_records
        return records

    def clear_meters(self):
        for name in self.meters.keys():
            self.clear_meter(name)

    # ------ update ------

    def update(self, cur_iter, total_iter, time, losses, lr, mode='train'):
        '''write current info into the log file
        cur_iter: int
        total_iter: int
        time: float or time.time() instance
        losses: an OrderedDict instance that records loss name (key) and loss val (value)
        lr: float
        '''
        msg = '[%5s] ' % mode # typical: train, val, test
        msg += 'iter: [%d/%d]\t' % ( cur_iter, total_iter )
        msg += 'time %.3f\t' % ( time )
        for kk in losses.keys():
            msg += '%s: %2.3f\t' % (kk, losses[kk])
        msg += 'lr %.6f\t' % ( lr )
        print( msg )
        with open(self.log_path, "a") as log_file:
            log_file.write('%s\n' % msg)