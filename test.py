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
from constant_autoregression.dataset.load_dataset import Input_Batch, Output_Batch, no_of_output_space
from constant_autoregression.util import LpLoss, Printer, get_time, count_params, set_seed, return_checkpoint, dynamic_weight_loss, dynamic_weight_loss_sq, create_current_results_folder, load_auguments, save_config_file, create_data, create_next_data




p = Printer(n_digits=6)
args = arg_parse()
args = load_auguments(args, "arguments")



def roll_out_test_during_training( args, model, loader, timestamps):
    model.eval()
    t_pass_test = args.next_input_time_stamps
    input_range = args.input_time_stamps
    total_range = args.total_t_range
    no_of_input = args.input_time_stamps
    t_pred_steps = args.output_time_stamps
    timestamps = torch.tensor(timestamps).to(device)
    with torch.no_grad():

        for b, (u_base, u_super, x, variables) in enumerate(loader):   #change to tesst loader

            test_data = u_base.permute(0,2,1).float().to(device)
            rand_input_type_test = 2
            input_batch = Input_Batch(data=test_data, input_sample_type=rand_input_type_test, input_range=input_range, total_range = total_range,  no_of_input=no_of_input, dt_input=1)

            rand_output_type_test = 2

            random_array_y = [
                Output_Batch(input_batch.input_indicies, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=200), 
            ]

            #import pdb; pdb.set_trace()

            for samp in range(len(random_array_y)):

                if rand_output_type_test == 2:
                        no_of_output_test = random_array_y[samp].output_indicies.shape[0]


                x = input_batch.input
                y = random_array_y[samp].output
                time_indicies = torch.cat((input_batch.input_indicies, random_array_y[samp].output_indicies), dim=-1)

                xy = torch.cat((x,y), dim=-1)
                xy_t = torch.ones_like(xy).to(device)


                #import pdb; pdb.set_trace()
                xy_t = xy_t*timestamps[time_indicies]


                #no_of_output_test = random_array_y[samp].output_indicies.shape[0]

                #output_indices_test = random_array_y[samp]

                #time_indicies_test = torch.cat((input_indicies_test.long(), output_indices_test.long() ), dim=-1)
                
                #import pdb; pdb.set_trace()
                #xy = test_data[ ..., time_indicies ]

                if b == 0:
                    train_actual = xy[..., t_pred_steps:].clone()
                else:
                    train_actual = torch.cat((train_actual, xy[..., t_pred_steps:]), dim=0)
            

                xy_t = torch.ones_like((xy)).to(device)
                xy_t = xy_t*timestamps[time_indicies]

                x = xy[..., : t_pred_steps]
                x_t = xy_t[..., :t_pred_steps]

                time_stamps = [i for i in range(0, len(time_indicies)+1, t_pass_test)]
                
                for t in range(len(time_stamps) - 2 ):
                    y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                    y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]

                    # print("\n")
                    # print("x -->", time_stamps[t], time_stamps[t+1] )
                    # print("x_t -->", x_t[0,0,:])
                    # print("y -->", time_stamps[t+1], time_stamps[t+2] )
                    # print("y_t -->", y_t[0,0,:])

                    if args.model_mode.startswith("constant_dt"):
                        out = model(x).to(device)

                    if args.model_mode.startswith("variable_dt"):
                        #model_operation = "add"
                        if args.model_input_operation.startswith("add"):
                            out = model( torch.cat((x, torch.zeros(x.shape).to(device)), dim = -1) + x_t ) # ADD
                        elif  args.model_input_operation.startswith("concat"):
                            out = model( torch.cat((x, x_t), dim=-1 ) ).to(device)  #CONCAT


                    if t == 0:
                        pred = out[...,:t_pass_test]
                    else:
                        pred = torch.cat((pred, out[...,:t_pass_test]), -1)

                    if (t_pass_test < t_pred_steps) and (t == no_of_output_test - t_pred_steps + 1 - 1):
                        pred = torch.cat((pred, out[...,t_pass_test: ]), -1)

                    if t < no_of_output_test-t_pred_steps:
                        x = torch.cat((x[..., t_pass_test:], out[...,:t_pass_test]), dim=-1)
                        x_t = torch.cat((x_t[..., t_pass_test:], y_t[..., :t_pass_test]), dim=-1)

                        
                if b == 0:
                    train_prediction = pred.clone()
                else:
                    train_prediction = torch.cat((train_prediction,pred), dim=0)

    test_l2_full = torch.mean((train_prediction-train_actual)**2, dim=[0,1] ).sum()

    return test_l2_full, train_prediction, train_actual






def test_only( args, model, loader, timestamps):
    model.eval()

    t_pass_test = args.next_input_time_stamps
    input_range = args.input_time_stamps
    total_range = args.total_t_range
    no_of_input = args.input_time_stamps
    t_pred_steps = args.output_time_stamps
    timestamps = torch.tensor(timestamps).to(device)
    with torch.no_grad():

        for b, (u_base, u_super, x, variables) in enumerate(loader):   #change to tesst loader

            test_data = u_base.permute(0,2,1).float().to(device)
            rand_input_type_test = 2
            input_batch = Input_Batch(data=test_data, input_sample_type=rand_input_type_test, input_range=input_range, total_range = total_range,  no_of_input=no_of_input, dt_input=1)

            rand_output_type_test = 2

            random_array_y = [
                Output_Batch(input_batch.input_indicies, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=200), 
            ]

            #import pdb; pdb.set_trace()

            for samp in range(len(random_array_y)):

                if rand_output_type_test == 2:
                        no_of_output_test = random_array_y[samp].output_indicies.shape[0]


                x = input_batch.input
                y = random_array_y[samp].output
                time_indicies = torch.cat((input_batch.input_indicies, random_array_y[samp].output_indicies), dim=-1)

                xy = torch.cat((x,y), dim=-1)
                xy_t = torch.ones_like(xy).to(device)


                #import pdb; pdb.set_trace()
                xy_t = xy_t*timestamps[time_indicies]


                #no_of_output_test = random_array_y[samp].output_indicies.shape[0]

                #output_indices_test = random_array_y[samp]

                #time_indicies_test = torch.cat((input_indicies_test.long(), output_indices_test.long() ), dim=-1)
                
                #import pdb; pdb.set_trace()
                #xy = test_data[ ..., time_indicies ]

                if b == 0:
                    train_actual = xy[..., t_pred_steps:].clone()
                else:
                    train_actual = torch.cat((train_actual, xy[..., t_pred_steps:]), dim=0)
            

                xy_t = torch.ones_like((xy)).to(device)
                xy_t = xy_t*timestamps[time_indicies]

                x = xy[..., : t_pred_steps]
                x_t = xy_t[..., :t_pred_steps]

                time_stamps = [i for i in range(0, len(time_indicies)+1, t_pass_test)]
                
                #import pdb; pdb.set_trace()
                for t in range(len(time_stamps) - 2 ):
                    
                    y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                    y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]

                    # print("\n")
                    # print("x -->", time_stamps[t], time_stamps[t+1] )
                    # print("x_t -->", x_t[0,0,:])
                    # print("y -->", time_stamps[t+1], time_stamps[t+2] )
                    # print("y_t -->", y_t[0,0,:])
                    #import pdb; pdb.set_trace()
                    if args.model_mode.startswith("constant_dt"):
                        out = model(x).to(device)

                    if args.model_mode.startswith("variable_dt"):
                        #model_operation = "add"
                        if args.model_input_operation.startswith("add"):
                            out = model( torch.cat((x, torch.zeros(x.shape).to(device)), dim = -1) + x_t ) # ADD
                        elif  args.model_input_operation.startswith("concat"):
                            out = model( torch.cat((x, x_t), dim=-1 ) ).to(device)  #CONCAT


                    if t == 0:
                        pred = out[...,:t_pass_test]
                    else:
                        pred = torch.cat((pred, out[...,:t_pass_test]), -1)

                    if (t_pass_test < t_pred_steps) and (t == no_of_output_test - t_pred_steps + 1 - 1):
                        pred = torch.cat((pred, out[...,t_pass_test: ]), -1)

                    if t < no_of_output_test-t_pred_steps:
                        x = torch.cat((x[..., t_pass_test:], out[...,:t_pass_test]), dim=-1)
                        x_t = torch.cat((x_t[..., t_pass_test:], y_t[..., :t_pass_test]), dim=-1)

                        
                if b == 0:
                    train_prediction = pred.clone()
                else:
                    train_prediction = torch.cat((train_prediction,pred), dim=0)

    test_l2_full = torch.mean((train_prediction-train_actual)**2 )

    return test_l2_full, train_prediction, train_actual


# def test_during_training(args, proto, times_eval):
#     file_saved = "protocol_" + str(proto) +".pt"
#     import pdb; pdb.set_trace()
#     #saved_result = os.path.join(args.current_result_save_path, _1.pt"
#     saved_result = torch.load(   os.path.join(args.current_result_save_path, file_saved  ),  map_location=device )   

#     epoch = saved_result["saved_epoch"][-1]["epoch"]
#     model = saved_result["saved_epoch"][-1]["model"]


#     if args.input_shape_type.startswith("concat"):
#         model_hyperparameters = {"modes": args.fno_modes, "width": args.fno_hidden_dim, "input_size": args.input_time_stamps, "output_size": args.output_time_stamps }
#     else:
#         raise TypeError("Specify the input shape type")
    
#     model_test =  FNO1d(model_hyperparameters["modes"], model_hyperparameters["width"], model_hyperparameters["input_size"], model_hyperparameters["output_size"]).to(device)
#     model_test.load_state_dict(model)
#     t_pred_steps = args.output_time_stamps
#     t_pass_test = args.output_time_stamps
#     no_of_input = args.input_time_stamps
#     testing_mode = "A_R"

#     # id_50 = torch.arange(50,250,4)
#     # id_50[-1]  = 249

#     #id_128 = torch.cat( (torch.arange(50,170,4), torch.arange(170,250,1)))
#     #id_128[-1]  = 249


#     #id_100 = torch.arange(51,250,2)


#     #input_indicies_test = torch.arange(0,50,5).to(device)
#     #input_indicies_test = torch.sort(torch.randint(0,50,(no_of_input,)))[0].to(device)  #
#     input_indicies_test = torch.arange(10).to(device)
#     #[id_100.to(device), id_128.to(device), torch.arange(50,250,1).to(device)]  #torch.arange(50,250,4).to(device)
#     #random_array_y =  [torch.arange(50,250,4).to(device), id_100.to(device), torch.arange(50,250,1).to(device)]

#     #test_l2_full = torch.zeros(len(random_array_y)).to(device)
    
#     random_array_y =  times_eval.copy()


#     # prediction_var_rand_in = []
#     # actual_var_rand_in = []

#     # prediction_var = []
#     # actual_var = []

#     for b,test_data in enumerate(train_loader):   #change to tesst loader
#         test_data = test_data[0].to(device)

#         rand_input_type_test = 2
#         #input_indicies_test = Input_Batch(data=test_data, input_sample_type=rand_input_type_test, input_range=input_range, total_range = total_range,  no_of_input=no_of_input, dt_input=1).input_indicies

#         rand_output_type_test = 2
#         # random_array_y = [
#         #     Output_Batch(input_indicies_test, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=50).output_indicies,
#         #     Output_Batch(input_indicies_test, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=100).output_indicies,
#         #     Output_Batch(input_indicies_test, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=200).output_indicies

#         # ]
#         # random_array_y = [
#         #     Output_Batch(input_indicies_test, data=test_data, output_sample_type=rand_output_type_test, total_range=total_range, no_of_output=200).output_indicies
#         # ]


#         for samp in range(len(random_array_y)):

#             if rand_output_type_test == 2:
#                     no_of_output_test = random_array_y[samp].shape[0]

#             #test_loss = 0
#             no_of_output_test = random_array_y[samp].shape[0]

#             output_indices_test = random_array_y[samp]
            
            
    
#             x = test_data[...,input_indicies_test]
#             y = test_data[...,output_indices_test]


#             if b == 0:
#                 train_actual = y.clone()
#             else:
#                 train_actual = torch.cat((train_actual,y), dim=0)


#             x_t = torch.ones((x.shape[0],x.shape[1], no_of_input + t_pred_steps)).to(device)
#             y_t = torch.ones((y.shape[0],y.shape[1], no_of_output_test - t_pred_steps)).to(device)


#             time_x = timestamps[input_indicies_test]
#             time_y = timestamps[output_indices_test]

#             x_t = x_t*torch.cat((time_x,time_y[:t_pred_steps]), dim=-1).to(device)
#             y_t = y_t*time_y[t_pred_steps:].to(device)

#             #print([i for i in range(0, no_of_output_test-t_pred_steps + 1, t_pass_test)])
#             for t in range(0, no_of_output_test - t_pred_steps + 1, t_pass_test):
#                 # print("\n")
#                 # print("t_range -->", range(t,t + t_pred_steps))
#                 y_true = y[..., t:t + t_pred_steps]


#                 #model_mode = "constant_dt"

#                 out = model_test( x ).to(device)


#                 if t == 0:
#                     pred = out[...,:t_pass_test]
#                 else:
#                     pred = torch.cat((pred, out[...,:t_pass_test]), -1)

#                 if (t_pass_test < t_pred_steps) and (t == no_of_output_test - t_pred_steps + 1 - 1):
#                     pred = torch.cat((pred, out[...,t_pass_test: ]), -1)

#                 if t < no_of_output_test-t_pred_steps:
#                     if testing_mode == "T_F":
#                         x = torch.cat((x[..., t_pass_test:], y_true[...,:t_pass_test]), dim=-1)
#                     elif testing_mode == "A_R":
#                         x = torch.cat((x[..., t_pass_test:], out[...,:t_pass_test]), dim=-1)
#                     else:
#                         raise TypeError("Choose training_mode: 'T_F' or 'A_R' ")

#                     x_t = torch.cat((x_t[..., t_pass_test:], y_t[..., t : t+t_pass_test]), dim=-1)

#             #print("pred -->", pred.shape)
                    
#             if b == 0:
#                 train_prediction = pred.clone()
#             else:
#                 train_prediction = torch.cat((train_prediction,pred), dim=0)

#     results = {"prediction" : train_prediction, "true" : train_actual, "time_eval" : times_eval}

#     #import pdb; pdb.set_trace()
#     torch.save(results, os.path.join(args.current_result_save_path, f"test_result_epoch_{epoch}.pt") )


