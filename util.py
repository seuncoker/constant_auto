from datetime import datetime
import torch
from termcolor import colored
import sys, os

import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import random
import operator
from functools import reduce
from functools import partial
from torch.nn import functional as F
import json
from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Set up seed."""
    if seed == -1:
        seed = None
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)



def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time
    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


class Printer(object):
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """
        
        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(self, item, tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=-1, precision="second", is_silent=False):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2], avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window+1,len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, 'yellow'))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))
    

p = Printer(n_digits=6)


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):

        num_examples = x.size()[0]
        res = x.size()[1]
        tsteps = x.size()[2]

        x = x.permute(2,1,0)
        y = y.permute(2,1,0)
        diff_norms = torch.norm(x.reshape(tsteps,-1) - y.reshape(tsteps,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(tsteps,-1), self.p, 1)



        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# def save_checkpoint(epoch, time_elapsed, loss, model, optimizer ):
#     state_dict = {
#         "epoch": epoch,
#         "time_elapsed": time_elapsed,
#         "loss": loss,
#         "model": model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#     }
#     torch.save(state_dict, f"checkpoint_{epoch}.pt")



def return_checkpoint(epoch, time_elapsed, loss, model, optimizer ):
    state_dict = {
        "epoch": epoch,
        "time_elapsed": time_elapsed,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    return state_dict





def dynamic_weight_loss(ep, epochs, const_val, max_forward_pass):
    tt = torch.arange(max_forward_pass).to(device)  # const = -0.5
    a = torch.exp(torch.tensor((-const_val*(max_forward_pass/epochs)*ep)).clone().detach())
    return torch.exp(-a*tt)


def dynamic_weight_loss_sq(ep, epochs, const_val, max_forward_pass, no_f_pass):
    tt = torch.arange(no_f_pass).to(device)
    tt = tt*(max_forward_pass//no_f_pass) # const = -0.5
    a = torch.exp(torch.tensor(-const_val*(max_forward_pass/epochs)*ep)).clone().detach()
    return torch.exp(-a*(tt**2))



def create_current_results_folder(args):
    # Define the desired folder name and path

    result_name = "result"
    folder_path =  args.current_dir_path 
    path_result = os.path.join(folder_path, result_name)
    args.result_save_path = path_result
    try:
        os.mkdir(path_result)
        p.print(f"Folder '{result_name}' created successfully!")
    except FileExistsError:
        p.print(f"Folder '{result_name}' already exists!")


    datedate = str(datetime.now().date() )
    folder_path =  args.result_save_path 
    path_date = os.path.join(folder_path, datedate)
    args.current_date_save_path = path_date
    try:
        os.mkdir(path_date)
        p.print(f"Folder '{datedate}' created successfully!")
    except FileExistsError:
        p.print(f"Folder '{datedate}' already exists!")


    result_name = args.experiment
    result_path = os.path.join(path_date, result_name)
    args.current_result_save_path = result_path
    try:
        os.mkdir(result_path)
        p.print(f"Folder '{result_name}' created successfully!")
    except FileExistsError:
        p.print(f"Folder '{result_name}' already exists!")



def save_config_file(args):
    filename_args = "config"
    with open(os.path.join(args.current_result_save_path, filename_args), 'w') as f:
        json.dump(vars(args), f, indent=4)



# Read the JSON file and extract arguments
def load_auguments(args, filename):
    try:
        with open(filename+".json", "r") as f:
            data = json.load(f)
            for key, value in data.items():
                if hasattr(args, key):
                    setattr(args, key, value)  # Set argument if it exists
    except FileNotFoundError:
        p.print("Warning: 'arguments.json' not found. Using default arguments.")
    
    return args



def save_checkpoint(epoch, time_elapsed, loss, model, optimizer, prediction, actual ):
    state_dict = {
        "epoch": epoch,
        "time_elapsed": time_elapsed,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "prediction": prediction, 
        "actual": actual,
    }
    torch.save(state_dict, f"checkpoint_{epoch}.pt")


def return_checkpoint(epoch, time_elapsed, loss, model, optimizer, prediction, actual ):
    state_dict = {
        "epoch": epoch,
        "time_elapsed": time_elapsed,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "prediction": prediction, 
        "actual": actual,
    }
    return state_dict


def initialize_weights_xavier_uniform(model):
  for m in model.modules():
    if isinstance(m, nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)



def create_data(xy, xy_t, random_steps, t_pred_steps, horizon):

    x = torch.Tensor().to(device)
    x_t = torch.Tensor().to(device)
    y = torch.Tensor().to(device)
    y_t = torch.Tensor().to(device)

    for (ii, jj, kk) in zip(xy, xy_t, random_steps):
      xx = ii[..., kk - t_pred_steps : kk]
      xx_t = jj[..., kk - t_pred_steps : kk]

      yy = ii[..., kk : kk + (horizon*t_pred_steps)]
      yy_t = jj[..., kk : kk + (horizon*t_pred_steps)]

      x = torch.cat((x, xx[None,:]), 0).to(device)
      x_t = torch.cat((x_t, xx_t[None,:]), 0).to(device)

      y = torch.cat((y, yy[None,:]), 0).to(device)
      y_t = torch.cat((y_t, yy_t[None,:]), 0).to(device)

    return torch.cat((x,y),dim=-1), torch.cat((x_t, y_t),dim=-1)


def create_next_data(x, x_t, out, y, y_t, t_pass_train):
    #print("x_concat_out -->",x[..., t_pass_train:].shape, out[...,:t_pass_train].shape)
    x = torch.cat((x[..., t_pass_train:], out[...,:t_pass_train]), dim=-1)
    x_t = torch.cat((x_t[..., t_pass_train:], y_t[...,:t_pass_train]), dim=-1)
    return x, x_t