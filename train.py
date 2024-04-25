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
from constant_autoregression.test import roll_out_test_during_training



p = Printer(n_digits=6)
args = arg_parse()
args = load_auguments(args, "arguments")









def training_loop(
    args,
    count_t_iter,
    proto,
    ep,
    epochs,
    time_stamps,
    t_iteration,
    output_tray,
    last_epoch_no,
    rand_input_type,
    input_tray_spacing,
    rand_output_type,
    model,
    scheduler,
    optimizer,
    train_loader,
    criterion,

    no_of_input,
    t_pass_train,
    t_pred_steps,
    total_range, 
    input_range,
    timestamps,

    f_pass_weights,
    t_step_weights,

    model_mode,
    model_input_operation,

    horizon,
    horizon_grad,

    ):


    train_l2_full = 0
    # p.print(f"horizon: {horizon}")
    # p.print(f"t_iteration: {t_iteration}")


    for s in range(t_iteration):
        count_t_iter += 1
        # print("\n")
        # print(f"T_iter: {s}")
        #import pdb; pdb.set_trace()
        for (u_base, u_super, x, variables) in train_loader:
            #count += 1
            data = u_base.permute(0,2,1).float().to(device)

            for out_samp in range(len(output_tray)):
                no_of_output = output_tray[out_samp]

                input_batch = Input_Batch(data=data, input_sample_type=rand_input_type[s], input_range=input_range, total_range = total_range,  no_of_input=no_of_input, dt_input= input_tray_spacing[s])
                output_batch = Output_Batch(input_batch.input_indicies, data=data, output_sample_type=rand_output_type[s], total_range=total_range, no_of_output=output_tray[out_samp])

                if rand_output_type[s] == 2:
                    no_of_output = output_batch.output_indicies.shape[0]

                #import pdb; pdb.set_trace()
                # if ep == last_epoch_no+1:
                #     f_pass_weights = torch.ones(  no_of_output//t_pred_steps ).to(device)
                #     t_step_weights =  torch.ones( t_pred_steps ).to(device)


                x = input_batch.input
                y = output_batch.output
                time_indicies = torch.cat((input_batch.input_indicies, output_batch.output_indicies), dim=-1)

                xy = torch.cat((x,y), dim=-1)
                xy_t = torch.ones_like(xy).to(device)

                xy_t = xy_t*timestamps[time_indicies]

                time_stamps = [i for i in range(0, len(time_indicies)+1, t_pass_train)]


                if args.time_sampling_prediction == "random":
                    init_time_stamp_range = torch.tensor([t for t in range(t_pred_steps, total_range -  (t_pred_steps * horizon) + 1)]).to(device)
                    random_steps = init_time_stamp_range[torch.randint(len(init_time_stamp_range), (args.batch_size_train,))]
                    #random_steps = torch.tensor(init_time_stamp_range[ torch.randperm(len(init_time_stamp_range))[:args.batch_size_train] ] ).to(device)
                    #import pdb; pdb.set_trace()
                    # if ep == (last_epoch_no+1):
                    #     p.print(f"random_samples --> {random_steps}" )
                    xy, xy_t = create_data(xy, xy_t, random_steps, t_pred_steps, horizon)
                    x = xy[..., time_stamps[0]:time_stamps[1] ]
                    x_t = xy_t[..., time_stamps[0]:time_stamps[1] ]

                elif args.time_sampling_prediction == "constant":
                    #xy, xy_t = xy[...,:(horizon+1)*t_pred_steps], xy_t[:(horizon+1)*t_pred_steps],
                    x = xy[..., time_stamps[s]:time_stamps[s+1] ]
                    x_t = xy_t[..., time_stamps[s]:time_stamps[s+1] ]
                
                else:
                    raise TypeError("Specify time sampling predition")
                

                loss = 0
                a_l = 0



                with torch.no_grad():
                    
                    for t in range(horizon-horizon_grad):

                        if args.time_sampling_prediction == "constant":
                            y = xy[..., time_stamps[s+t+1]:time_stamps[s+t+2]]
                            y_t = xy_t[..., time_stamps[s+t+1]:time_stamps[s+t+2]]

                        elif args.time_sampling_prediction == "random":
                            y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                            y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]

                        #import pdb; pdb.set_trace()

                        if model_mode.startswith("constant_dt"):
                            #import pdb; pdb.set_trace()
                            out = model(x).to(device)

                        if model_mode.startswith("variable_dt"):
                            if model_input_operation.startswith("add"):
                                out = model( torch.cat((x, torch.zeros(x.shape).to(device)), dim = -1) + x_t ) # ADD
                            elif model_input_operation.startswith("concat"):
                                out = model( torch.cat((x, x_t), dim=-1 ) ).to(device)  #CONCAT
                        

                        # if ep == (last_epoch_no+1) :
                        #     p.print("\n")
                        #     if args.time_sampling_prediction == "constant":
                        #         p.print(f"x --> {time_stamps[s+t],time_stamps[s+t+1]}" )
                        #         p.print(f"x_t --> {x_t[:3,0,:5]}")
                        #         p.print(f"y --> {time_stamps[s+t+1],time_stamps[s+t+2]} ")
                        #         p.print(f"y_t --> {y_t[:3,0,:5]} ")
                        #         p.print(f"f_pass_weights[{a_l}] ->, {f_pass_weights[a_l]}")
                        #         p.print(f"loss ->, {loss}")
                        #     elif args.time_sampling_prediction == "random":
                        #         p.print(f"x --> {time_stamps[t],time_stamps[t+1]}" )
                        #         p.print(f"x_t --> {x_t[:3,0,:5]}")
                        #         p.print(f"y --> {time_stamps[t+1],time_stamps[t+2]} ")
                        #         p.print(f"y_t --> {y_t[:3,0,:5]} ")
                        #         p.print(f"f_pass_weights[{a_l}] ->, {f_pass_weights[a_l]}")
                        #         p.print(f"loss ->, {loss}")

                        a_l += 1

                        if t < no_of_output - t_pred_steps:
                            x = torch.cat((x[..., t_pass_train:], out[...,:t_pass_train]), dim=-1)
                            x_t = torch.cat((x_t[..., t_pass_train:], y_t[...,:t_pass_train]), dim=-1)
       
                            #x, x_t = create_next_data(x, x_t, out, y, y_t, t_pass_train)
                        
                        #print("loss->", loss)


                #import pdb; pdb.set_trace()
                for t in range(horizon-horizon_grad, horizon):
                    #import pdb; pdb.set_trace()

                    if args.time_sampling_prediction == "constant":
                        y = xy[..., time_stamps[s+t+1]:time_stamps[s+t+2]]
                        y_t = xy_t[..., time_stamps[s+t+1]:time_stamps[s+t+2]]

                    elif args.time_sampling_prediction == "random":
                        y = xy[..., time_stamps[t+1]:time_stamps[t+2]]
                        y_t = xy_t[..., time_stamps[t+1]:time_stamps[t+2]]
                    
                    #import pdb; pdb.set_trace()


                    if model_mode.startswith("constant_dt"):
                        #import pdb; pdb.set_trace()
                        out = model(x).to(device)


                    if model_mode.startswith("variable_dt"):
                        if model_input_operation.startswith("add"):
                            out = model( torch.cat((x, torch.zeros(x.shape).to(device)), dim = -1) + x_t ) # ADD
                        elif model_input_operation.startswith("concat"):
                            out = model( torch.cat((x, x_t), dim=-1 ) ).to(device)  #CONCAT


                    #import pdb; pdb.set_trace()
                    #loss_t = criterion(out, y).to(device)   ################ CHANGE
                    loss_t = criterion(out, y).sum(dim=[0,1]).to(device)
                    loss_t = torch.sqrt(loss_t)

                    loss_t_w = t_step_weights*loss_t

                    loss += f_pass_weights[a_l]*loss_t_w.sum()


                    #if ep == (last_epoch_no+1) :
                        #p.print("\n")
                        # if args.time_sampling_prediction == "constant":
                        #     p.print(f"x --> {time_stamps[s+t],time_stamps[s+t+1]}" )
                        #     p.print(f"x_t --> {x_t[:3,0,:5]}")
                        #     p.print(f"y --> {time_stamps[s+t+1],time_stamps[s+t+2]} ")
                        #     p.print(f"y_t --> {y_t[:3,0,:5]} ")
                        #     p.print(f"f_pass_weights[{a_l}] ->, {f_pass_weights[a_l]}")
                        #     p.print(f"loss ->, {loss}")
                        # elif args.time_sampling_prediction == "random":
                        #     p.print(f"x --> {time_stamps[t],time_stamps[t+1]}" )
                        #     p.print(f"x_t --> {x_t[:3,0,:5]}")
                        #     p.print(f"y --> {time_stamps[t+1],time_stamps[t+2]} ")
                        #     p.print(f"y_t --> {y_t[:3,0,:5]} ")
                        #     p.print(f"f_pass_weights[{a_l}] ->, {f_pass_weights[a_l]}")
                        #     p.print(f"loss ->, {loss}")

                    a_l += 1


                    if t < no_of_output - t_pred_steps:
                        x = torch.cat((x[..., t_pass_train:], out[...,:t_pass_train]), dim=-1)
                        x_t = torch.cat((x_t[..., t_pass_train:], y_t[...,:t_pass_train]), dim=-1)
                train_l2_full += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #import pdb; pdb.set_trace()
        if (count_t_iter) % 50 == 0:
            train_l2_error_err, prediction_train, actual_train = roll_out_test_during_training(args, model, train_loader, args.time_stamps)
            p.print(f"t_iter: {count_t_iter},  train_error: {train_l2_error_err:.4f}")

    return train_l2_full, model, count_t_iter
    





def training_protocol(
                    proto,
                    args,
                    epochs,
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    test_loader,
                    saved_epochs,
                    all_epoch_errors,
                    criterion,
                    ):

    assert(args.horizon <= (args.total_t_range - args.output_time_stamps)//args.output_time_stamps )
    model.train()

    #import pdb; pdb.set_trace()

    output_tray = no_of_output_space(output_space_type=3, predefined = torch.tensor([200]) ).output_tray    
    p.print(f"number of output samples: {output_tray}")


    last_epoch_no = 0
    results = []
    last_result = []
    error = torch.zeros(epochs,6).to(device)


    if saved_epochs != None:
        last_epoch_no = saved_epochs[-1]["epoch"]
        results = saved_epochs
        error = torch.cat((all_epoch_errors, error), dim=0)


    time_steps = [i for i in range(0, args.input_tspace_range+output_tray[0]+1, args.next_input_time_stamps)]   #change indexing of output_tray
    
    p.print(f"total horizon: {args.horizon}")
    p.print(f"last number of horizon with gradient: {args.horizon_grad}")




    if args.time_sampling_prediction == "constant":
        t_iteration = len(time_steps)-args.horizon - 1
    elif args.time_sampling_prediction == "random":
        t_iteration =  args.total_t_range
    else:
        raise TypeError("Specify time_sampling_prediciton")
    
    p.print(f"t_iteration: {t_iteration}")

    #import pdb; pdb.set_trace()
    rand_input_type = torch.tensor(args.input_sampler_type[proto])[torch.randint(0,len(args.input_sampler_type[proto]),(t_iteration,))]
    input_tray_spacing= torch.tensor(args.input_sampler_type_dt[proto])[torch.randint(0,len(args.input_sampler_type_dt[proto]),(t_iteration,))]
    rand_output_type = torch.tensor(args.output_sampler_type[proto])[torch.randint(0,len(args.output_sampler_type[proto]),(t_iteration,))]

    p.print(f"{len(rand_input_type)},{rand_input_type[:10]}" )
    p.print(f"{len(input_tray_spacing)},{input_tray_spacing[:10]}" ) 
    p.print(f"{len(rand_output_type)},{rand_output_type[:10]}" ) 


    f_pass_weights = torch.ones( args.horizon ).to(device)
    t_step_weights =  torch.ones( args.output_time_stamps ).to(device)

    count_t_iter = 0

    for ep in range(last_epoch_no+1, last_epoch_no + epochs + 1):


        if args.dynamic_loss_weight_per_fpass[proto]:
            f_pass_weights = dynamic_weight_loss_sq(ep-last_epoch_no, epochs, args.dynamic_loss_weight_per_fpass_constant_parameter[proto], (args.total_t_range - args.output_time_stamps)//args.output_time_stamps, args.horizon).to(device)
        elif args.dynamic_loss_weight_per_fpass[proto] == None:
            raise TypeError("Specify dynamic_w_l_f_pass (True or False) ")

        if args.dynamic_loss_weight_per_fpass_reversed[proto]:
            f_pass_weights = dynamic_weight_loss_sq(epochs-(ep-last_epoch_no)+1, epochs, (args.total_t_range - args.output_time_stamps)//args.output_time_stamps, args.horizon).to(device)
            f_pass_weights = torch.flip(f_pass_weights, dims=[0])
        elif args.dynamic_loss_weight_per_fpass_reversed[proto] == None:
            raise TypeError("Specify dynamic_w_l_f_pass_reversed (True or False) ")

        if args.dynamic_loss_weight_per_tstamp[proto]:
            t_step_weights = dynamic_weight_loss(ep, epochs, args.dynamic_loss_weight_per_tstamp_constant_parameter[proto],  args.output_time_stamps  ).to(device)
        elif args.dynamic_loss_weight_per_tstamp[proto] == None:
            raise TypeError("Specify dynamic_w_l_t_steps (True or False) ")


        # print("t_step_weights -->",t_step_weights)
        # print("f_pass_weights -->", f_pass_weights)

        t1 = default_timer()


        #import pdb; pdb.set_trace()
        train_l2_full, model, count_t_iter = training_loop(
                        args,
                        count_t_iter,
                        proto,
                        ep,
                        time_stamps = time_steps,
                        epochs = epochs,
                        last_epoch_no = last_epoch_no,
                        rand_input_type = rand_input_type,
                        input_tray_spacing= input_tray_spacing,
                        rand_output_type = rand_output_type,
                        model = model,
                        output_tray = output_tray,
                        t_iteration = t_iteration,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        train_loader = train_loader,
                        criterion = criterion,


                        no_of_input =  args.input_time_stamps,
                        t_pass_train = args.next_input_time_stamps,
                        t_pred_steps = args.output_time_stamps,

                        total_range = args.total_t_range, #250,   #not defiend yett
                        input_range = args.input_tspace_range,  #not defiend yett
                        timestamps = torch.tensor(args.time_stamps).to(device),

                        f_pass_weights = f_pass_weights,
                        t_step_weights = t_step_weights,

                        model_mode = args.model_mode,
                        model_input_operation = args.model_input_operation,

                        horizon = args.horizon,
                        horizon_grad = args.horizon_grad,

        )

        current_lr = optimizer.param_groups[0]['lr']
        train_l2_full = train_l2_full/ (len(output_tray)*t_iteration*args.n_train)
        
        t2 = default_timer()
        scheduler.step()

        #import pdb; pdb.set_trace()
        train_l2_error, prediction_train, actual_train = roll_out_test_during_training(args, model, train_loader, args.time_stamps)
        test_l2_error, prediction_test, actual_test = roll_out_test_during_training(args, model, test_loader, args.time_stamps)

        errs_statss = torch.tensor([ ep, t2 - t1, current_lr, train_l2_full, train_l2_error, test_l2_error]).to(device)
        error[ep-1,:errs_statss.shape[-1]] = errs_statss
        p.print(f"ep: {ep}, time: {(t2-t1):.4f}, lr: {current_lr}, train: {train_l2_full:.4f}, train_error: {train_l2_error:.4f} test_error: {test_l2_error:.4f}")
        
        #import pdb; pdb.set_trace()
        if (ep) % args.epoch_save_interval == 0:
            results.append(return_checkpoint(ep, errs_statss[1], errs_statss[2:], model, optimizer,prediction_train, actual_train))

        #epoch_count += 1
    
    prediction = prediction_train
    actual = actual_train
    #import pdb; pdb.set_trace()
    results.append(return_checkpoint(ep, errs_statss[1], errs_statss[2:], model, optimizer, prediction, actual))
    torch.save({"saved_epoch": results, "all_error":error }, os.path.join(args.current_result_save_path, f"protocol_{proto+1}.pt"))
    

    if proto == args.number_of_training_protocol - 1:
        torch.save(error, os.path.join(args.current_result_save_path, f"errors.pt") )
    
    last_result.append(return_checkpoint(ep, errs_statss[1], errs_statss[2:], model, optimizer, prediction, actual))
    
    #import pdb; pdb.set_trace()
    
    return {"saved_epoch": last_result, "all_error": error}

