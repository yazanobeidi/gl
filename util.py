import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from tqdm.utils import _screen_shape_wrapper
from tqdm import tqdm
import sys
import shutil
import os

def set_random_seed(seed):
    import numpy as np
    import random
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_daytime():
    time = datetime.datetime.now()
    day, time = time.strftime("%Y%m%d"), time.strftime("%H%M%S")
    return day, time

def get_tensorboard(run_description, experiment, basepath):
    day, time = get_daytime()
    path_to_tb = f'{basepath}/{time}/{day}_{run_description}/'
    print(f'Loaded SummaryWriter at {path_to_tb}')
    return SummaryWriter(path_to_tb)

def unfitted(o):
    try:
        check_is_fitted(o)
    except NotFittedError:
        return True
    except:
        print('unknown exception')
        raise
    return False

class multiline_tqdm(tqdm):
    #fixes https://github.com/tqdm/tqdm/issues/630#issuecomment-1321245383
    def __init__(self, *args, desc="", **kwargs):
        super().__init__(*args, **kwargs)
        self.subbar = None
        self.set_description(desc)
    def set_description(self, desc=None, refresh=True):
        screen_width, _ = _screen_shape_wrapper()(sys.stdout)
        max_len = screen_width
        if len(desc) > max_len*.7:
            if not self.subbar:
                self.subbar = multiline_tqdm(range(len(self)))
                self.subbar.n = self.n
                self.default_bar_format = self.bar_format
                self.bar_format = "{desc}"
            super().set_description_str(desc=desc[:screen_width], refresh=refresh)
            self.subbar.set_description(desc[screen_width:])
        else:
            if self.subbar:
                self.bar_format = self.default_bar_format
                self.subbar.leave = False
                self.subbar.close()
            super().set_description(desc=desc, refresh=refresh)
    def update(self, n=1):
        if self.subbar:
            self.subbar.update(n)
            self.last_print_n = self.subbar.last_print_n
            self.n = self.subbar.n
        else:
            super().update(n)
    def close(self):
        if self.subbar:
            self.subbar.leave = self.leave
            self.subbar.close()
        super().close()

def persist_run_impl(basepath):
    shutil.copytree(os.getcwd(), os.path.join(basepath, 'code_archive'), 
        ignore=shutil.ignore_patterns("data", "notebooks", ".git", "log", 
                ".ipynb_checkpoints", "modelarchive", "historic_logs", "oldlogs"))

def import_model_loader(model_filename, experiment='cover'):
    fname = model_filename.replace('/','').replace('.','')
    print(f'loading model {experiment}.models.{fname}')
    # for compatibility with older code
    loader = 'load_covertype_models' if experiment == 'cover' else 'load_models'
    return getattr(__import__(f'{experiment}.models.{fname}', 
                            fromlist=[loader]), 
                  loader)