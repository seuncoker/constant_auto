import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt


from termcolor import colored
import sys, os
from datetime import datetime

import operator
from functools import reduce
from functools import partial
from timeit import default_timer

import csv
import h5py

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from constant_autoregression.argparser import arg_parse 

from constant_autoregression.dataset.load_dataset import load_data, load_dataset_E1
from constant_autoregression.util import LpLoss, Printer, get_time, count_params, set_seed, return_checkpoint, dynamic_weight_loss, dynamic_weight_loss_sq, create_current_results_folder, load_auguments, save_config_file
from constant_autoregression.model import FNO1d, load_model, get_model
from constant_autoregression.train import training_protocol
from constant_autoregression.test import test_only

#import pdb; pdb.set_trace()

################################################################
# load arguments
################################################################
p = Printer(n_digits=6)
args = arg_parse()
args = load_auguments(args, "arguments")



################################################################
# load seed
################################################################

set_seed(args.seed)


################################################################
# Create result files
################################################################

create_result_folder = True
if create_result_folder:
    create_current_results_folder(args)


################################################################
# Load training data
################################################################
#train_loader, val_loader, test_loader = load_data(args)
train_loader, val_loader, test_loader = load_dataset_E1(args)



p.print(f"Minibatches for train: {len(train_loader)}")
p.print(f"Minibatches for val: {len(val_loader)}")
p.print(f"Minibatches for test: {len(test_loader)}")



################################################################
# Model 
################################################################
#import pdb; pdb.set_trace()
if args.new_training == False:
    assert args.load_experiment != None
    
    load_filename =  torch.load(args.pretrained_model, map_location=device)
    saved_epochs = load_filename["saved_epoch"].copy()
    all_epoch_errors = load_filename["all_error"].clone()
    last_model_dict = saved_epochs[-1]["model"]
    model = load_model(args, last_model_dict, device)
    p.print(f"Continue Training using {load_filename}")
    
else:
    saved_epochs = None
    all_epoch_errors = None
    p.print(f"New Training")
    model = get_model(args, device)
    


p.print(f"model {model}")

#import pdb; pdb.set_trace()


################################################################
# Sort arguments for different training protocol
################################################################

args.number_of_training_protocol = len(args.training_protocols)
key_names = list(args.training_protocols[0].keys())
for k in range(len(key_names) ):
    key_name = key_names[k]
    values = [d[key_name] for d in args.training_protocols]
    setattr(args, key_name, values)

################################################################
# Save Configuration file
################################################################
#import pdb; pdb.set_trace()
save_config_file(args)


################################################################
# Training Protocol
################################################################

if args.mode.startswith("train"):
    for proto in range(args.number_of_training_protocol):
        # training_parameter = {"initialise_optimiser": args.initialise_optimiser[proto],
        #                     "optimizer_type":args.optimiser_type[proto],
        #                     "sheduler_type": args.sheduler_type[proto],
        #                     "learning_rate": args.learning_rate[proto],
        #                     "sheduler_step": args.sheduler_step[proto],
        #                     "sheduler_gamma": args.sheduler_gamma[proto],
        #                     "weight_decay": args.weight_decay[proto]
        #                     }
        #import pdb; pdb.set_trace()

        if args.optimiser_type[proto].startswith("adam"):
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[proto], weight_decay=args.weight_decay[proto])
        if args.sheduler_type[proto].startswith("steplr"):
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.sheduler_step[proto], gamma=args.sheduler_gamma[proto])

        if args.new_training == False and args.initialise_optimiser[proto] == False:
            p.print("Loading optimizer last state")
            optimizer.load_state_dict(saved_epochs[-1]["optimizer"])
        else:
            pass

        # if  proto > 0 and args.initialise_optimiser[proto] == False:
        #     p.print("Loading optimizer last state")
        #     optimizer.load_state_dict(saved_epochs[-1]["optimizer"])
        # else:
        #     pass

        if args.loss_train_type.startswith("l2"):
            criterion = LpLoss(size_average=False, reduction=False)
        elif args.loss_train_type.startswith("mse"):
            criterion = torch.nn.MSELoss(reduction="none")
        else:
            pass
        
        epochs = args.epochs[proto]

        #import pdb; pdb.set_trace()

        saved_results = training_protocol(
            proto = proto,
            args = args,
            epochs = epochs,
            model = model,
            optimizer = optimizer,
            scheduler = scheduler,
            train_loader = train_loader,
            test_loader = test_loader,
            saved_epochs = saved_epochs,
            all_epoch_errors = all_epoch_errors,
            criterion = criterion,
            )
        
        saved_epochs = saved_results["saved_epoch"].copy()
        all_epoch_errors = saved_results["all_error"].clone()
        p.print(f"End of Training Protocol {proto+1} ..........")   


elif args.mode.startswith("test"):
    import pdb; pdb.set_trace()
    current_result_save_path = args.current_result_save_path
    test_only_path = args.test_only_path
    arguments_file = os.path.join(args.test_only_path, "config" )
    args = load_auguments(args, arguments_file)

    args.current_result_save_path = current_result_save_path
    args.test_only_path = test_only_path

    file_saved = "protocol_" + str(args.test_only_protocol_no) +".pt"
    saved_result = torch.load(   os.path.join(args.test_only_path, file_saved  ),  map_location=device )   
    epoch = saved_result["saved_epoch"][args.test_only_epoch_index]["epoch"]
    last_model_dict = saved_result["saved_epoch"][args.test_only_protocol_no]["model"]
    model = load_model(args, last_model_dict, device)
    test_l2_full, train_prediction, train_actual = test_only( args, model, train_loader, args.time_stamps)
    result = {"prediction" : train_prediction, "true" : train_actual, "time_stamps" : args.time_stamps}
    p.print(f"test: {test_l2_full}")
    torch.save(result, os.path.join(args.current_result_save_path, f"test_result_epoch_{epoch}.pt") )



    
# for proto in range(args.number_of_training_protocol):
#     print("proto-->", proto)
#     training_parameter = {"initialise_optimiser": args.initialise_optimiser[proto],
#                         "optimizer_type":args.optimiser_type[proto],
#                         "sheduler_type": args.sheduler_type[proto],
#                         "learning_rate": args.learning_rate[proto],
#                         "sheduler_step": args.sheduler_step[proto],
#                         "sheduler_gamma": args.sheduler_gamma[proto],
#                         "weight_decay": args.weight_decay[proto]
#                         }

    # if proto > 0:
    #     new_training = False
    #     saved_epochs = saved_results["saved_epoch"].copy()
    #     all_epoch_errors = saved_results["all_error"].clone()

    # if proto == 0:
    #         if args.pretrained_model == None:
    #                 new_training = True
    #                 saved_epochs = None
    #                 all_epoch_errors = None

    #         elif args.pretrained_model != None:
    #                 new_training = False
    #                 pretrained = torch.load(args.pretrained_model, map_location=device)
    #                 saved_epochs = pretrained["saved_epoch"].copy()
    #                 all_epoch_errors = pretrained["all_error"].clone()
    
# training_parameter = {"initialise_optimiser": args.initialise_optimiser[proto],
#                         "optimizer_type":args.optimiser_type[proto],
#                         "sheduler_type": args.sheduler_type[proto],
#                         "learning_rate": args.learning_rate[proto],
#                         "sheduler_step": args.sheduler_step[proto],
#                         "sheduler_gamma": args.sheduler_gamma[proto],
#                         "weight_decay": args.weight_decay[proto]
#                         }
# model = FNO1d(model_hyperparameters["modes"], model_hyperparameters["width"], model_hyperparameters["input_size"], model_hyperparameters["output_size"]).to(device)

# initialise_optimiser = model_training_parameters["initialise_optimiser"]
# optimiser_type = model_training_parameters["optimizer_type"]
# sheduler_type = model_training_parameters["sheduler_type"]
# learning_rate = model_training_parameters["learning_rate"]
# scheduler_step = model_training_parameters["sheduler_step"]
# scheduler_gamma = model_training_parameters["sheduler_gamma"]
# weight_decay = model_training_parameters["weight_decay"]

# if optimiser_type.startswith("adam"):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# if sheduler_type.startswith("steplr"):
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
# if args.loss_train_type.startswith("l2"):
#     myloss = LpLoss(size_average=False, reduction=False)


# last_epoch_no = 0
# results = []
# last_result = []
# error = torch.zeros(epochs,6).to(device)
# #import pdb; pdb.set_trace()
# # PRE TRAINED MODEL
# if new_training == False:
# assert saved_epochs != None
# model.load_state_dict(saved_epochs[-1]["model"])
# last_epoch_no = saved_epochs[-1]["epoch"]
# results = saved_epochs
# error = torch.cat((all_epoch_errors, error), dim=0)

# if initialise_optimiser:
# pass
# else:
# optimizer.load_state_dict(saved_epochs[-1]["optimizer"])


# def training_protocol(
#                     proto,
#                     epochs,
#                     new_training,
#                     training_parameter,
#                     saved_epochs,
#                     all_epoch_errors,
#                     dynamic_w_l_f_pass,
#                     dynamic_w_l_f_pass_reversed,
#                     dynamic_w_l_t_steps,
#                     dynamic_w_l_f_pass_constant,
#                     dynamic_w_l_t_steps_constant,
#                     input_sampler_type,
#                     input_sampler_type_dt,
#                     output_sampler_type,
#                     ):
#     #training_parameter = {"lr": args.learning_rate[proto], "s_step": args.shed_step[proto], "s_gamma": args.shed_gamma[proto], "weight_decay": args.weight_decay[proto] }
    
#     # if args.input_shape_type.startswith("concat"):
#     #     model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": 2*args.input_time_stamps + 1, "output_size": args.output_time_stamps }
#     # else:
#     #      raise TypeError("Specify the input shape type")


#     if args.model_mode.startswith("constant_dt"):
#         model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps }
    
#     elif args.model_mode.startswith("variable_dt"):
#         if args.model_input_operation.startswith("add"):
#             model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps }
#         elif args.model_input_operation.startswith("concat"):
#             model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": 2*args.input_time_stamps + args.output_time_stamps, "output_size": args.output_time_stamps }
#     else:
#          raise TypeError("Specify the input shape type")
    
#     #import pdb; pdb.set_trace()
#     res = train_with_horizon(
#                     proto = proto,
#                     model_hyperparameters = model_hyperparameters,
#                     model_training_parameters = training_parameter,
#                     epochs=epochs,
                    
#                     no_of_input =  args.input_time_stamps,

#                     t_pass_train = args.next_input_time_stamps,
#                     training_mode = "A_R",
#                     t_pass_test = args.next_input_time_stamps,
#                     testing_mode = "A_R",

#                     t_pred_steps = args.output_time_stamps,

#                     #no_of_output_tsteps_low = args.output_tspace_low,
#                     #output_tsteps_sampler_type= args.output_tsteps_sampler_type,
#                     #output_tstamps_sampler_type = args.output_tstamps_sampler_type,

#                     input_range_multiple = args.no_input_tspace[proto], 

#                     total_range = timestamps.shape[-1], #250,   #not defiend yett
#                     input_range = args.input_tspace_range,  #not defiend yett

#                     new_training = new_training,
#                     saved_epochs = saved_epochs,
#                     all_epoch_errors = all_epoch_errors,
                    
#                     input_sampler_type = input_sampler_type,
#                     input_sampler_type_dt = input_sampler_type_dt,
#                     output_sampler_type = output_sampler_type,

#                     dynamic_w_l_f_pass = dynamic_w_l_f_pass,
#                     dynamic_w_l_f_pass_constant = dynamic_w_l_f_pass_constant,
#                     dynamic_w_l_f_pass_reversed = dynamic_w_l_f_pass_reversed,

#                     dynamic_w_l_t_steps = dynamic_w_l_t_steps,
#                     dynamic_w_l_t_steps_constant = dynamic_w_l_t_steps_constant,

#                     epoch_save_interval = args.epoch_save_interval,
#                     model_mode = args.model_mode,
#                     model_input_operation = args.model_input_operation,
#                     training_type = args.training_type,
#                     horizon = args.horizon,

#     )

#     return res




# def main():
    
#     #import pdb; pdb.set_trace()

#     #import pdb; pdb.set_trace()
#     args.number_of_training_protocol = len(args.training_protocols)
#     key_names = list(args.training_protocols[0].keys())
#     for k in range(len(key_names) ):
#         key_name = key_names[k]
#         values = [d[key_name] for d in args.training_protocols]
#         setattr(args, key_name, values)

#     #import pdb; pdb.set_trace()
#     save_config_file(args)
#     #import pdb; pdb.set_trace()
#     for proto in range(args.number_of_training_protocol):
#         print("proto-->", proto)
#         if proto == 0:
#                 if args.pretrained_model == None:
#                         new_training = True
#                         saved_epochs = None
#                         all_epoch_errors = None

#                 elif args.pretrained_model != None:
#                         new_training = False
#                         pretrained = torch.load(args.pretrained_model, map_location=device)
#                         saved_epochs = pretrained["saved_epoch"].copy()
#                         all_epoch_errors = pretrained["all_error"].clone()
        
#         #import pdb; pdb.set_trace()

#         if proto > 0:
#                 new_training = False
#                 saved_epochs = saved_results["saved_epoch"].copy()
#                 all_epoch_errors = saved_results["all_error"].clone()
        
#         #import pdb; pdb.set_trace()
#         training_parameter = {"initialise_optimiser": args.initialise_optimiser[proto],
#                                "optimizer_type":args.optimiser_type[proto],
#                                "sheduler_type": args.sheduler_type[proto],
#                                "learning_rate": args.learning_rate[proto],
#                                "sheduler_step": args.sheduler_step[proto],
#                                "sheduler_gamma": args.sheduler_gamma[proto],
#                                "weight_decay": args.weight_decay[proto]
#                                }

#         #import pdb; pdb.set_trace()
#         saved_results = training_protocol(
#                                         proto = proto,
#                                         epochs = args.epochs[proto],
#                                         new_training = new_training,
#                                         training_parameter = training_parameter,
#                                         saved_epochs = saved_epochs, 
#                                         all_epoch_errors = all_epoch_errors,
#                                         dynamic_w_l_f_pass =args.dynamic_loss_weight_per_fpass[proto],
#                                         dynamic_w_l_f_pass_reversed = args.dynamic_loss_weight_per_fpass_reversed[proto],
#                                         dynamic_w_l_t_steps =args.dynamic_loss_weight_per_tstamp[proto],
#                                         dynamic_w_l_f_pass_constant = args.dynamic_loss_weight_per_fpass_constant_parameter[proto],
#                                         dynamic_w_l_t_steps_constant = args.dynamic_loss_weight_per_tstamp_constant_parameter[proto],
#                                         input_sampler_type = torch.tensor(args.input_sampler_type[proto]),
#                                         input_sampler_type_dt = torch.tensor(args.input_sampler_type_dt[proto]),
#                                         output_sampler_type = torch.tensor(args.output_sampler_type[proto]),
#                                         )
#         p.print(f"End of Training Protocol {proto+1} ..........")
        #import pdb; pdb.set_trace()


