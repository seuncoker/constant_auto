import argparse
import sys, os
from pathlib import Path
from typing import List
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


def arg_parse():
    parser = argparse.ArgumentParser(description='constant_Autoregression argparse.')


    # DATASET
    parser.add_argument('--mode', type=str,
                        help=' Train or Test')
    
    parser.add_argument('--test_only_path', type=str,
                        help=' Test only file path')
    parser.add_argument('--test_only_protocol_no', type=int,
                        help=' Test only file path')  
    parser.add_argument('--test_only_epoch_index', type=int,
                        help=' Test only file path')  
       
    parser.add_argument('--dataset_name', type=str,
                        help=' Type of dataset. Choose from: "E1, E2, E3"')
    parser.add_argument('--experiment', type=str,
                        help=' Name of experiment ')
    parser.add_argument('--dataset_train_path', type=str,
                        help=' URL of train datasets"')
    parser.add_argument('--dataset_valid_path', type=str,
                        help=' URL of valid datasets"')
    parser.add_argument('--dataset_test_path', type=str,
                        help=' URL of test datasets"')
    parser.add_argument('--total_t_range', type=int,
                        help='Number of total time steps')
    parser.add_argument('--timestamps', type=list,
                        help='Number of total time steps')
                


    parser.add_argument('--n_train', type=int,
                        help='The first n_train examples will be used for the dataset. If -1, will use the full dataset.')
    parser.add_argument('--n_test', type=int,
                    help='The first n_test examples will be used for the dataset. If -1, will use the full dataset.')
    parser.add_argument('--batch_size_train', type=int,
                        help=' Training batch size.')
    parser.add_argument('--batch_size_test', type=int,
                        help=' Training batch size.')
    

    parser.add_argument('--root_dir', type=Path,
                        help='The direcotry of the project')
    parser.add_argument('--seed', type=int,
                        help='number of wells')
    
    
    ## MODEL

    parser.add_argument('--model_type', type=str,
                        help='FNO 1D')
    parser.add_argument('--fno_hidden_dim', type=int,
                        help='hidden dimesion of layers')
    parser.add_argument('--fno_hidden_layers', type=int,
                        help='number of layers')
    parser.add_argument('--fno_modes', type=int,
                        help='number of fourier modes')
    parser.add_argument('--input_shape_type', type=str,
                        help='type of input processing between x and  x_t')
    parser.add_argument('--model_mode', type=str,
                        help='constant or variiable model')
    parser.add_argument('--model_input_operation', type=str,
                        help='add or concatenate')
                

    # AUTOREGRESSION
    ## input
    parser.add_argument('--input_time_stamps', type=int,
                        help='number of input time stamps')
    parser.add_argument('--input_tspace_range', type=int,
                    help='range of the input sample range')
    parser.add_argument('--no_input_tspace', type=list,
                        help='number of input sample space: max_value = (total tsteps/input_tspace_range)')


    parser.add_argument('--input_sampler_type', type=list,
                        help='input type')
    parser.add_argument('--input_sampler_type_dt', type=list,
                    help='input_dt_time')
    parser.add_argument('--output_sampler_type', type=list,
                        help='output type')



    ## output
    parser.add_argument('--output_time_stamps', type=int,
                        help='number of output time stamps per forward pass')
    parser.add_argument('--output_tspace_low', type=int,
                    help='smallest number of output prediction stamps')
    parser.add_argument('--output_tsteps_sampler_type', type=int,
                    help='Determine the number of different time steps for each batch' )
    parser.add_argument('--output_tstamps_sampler_type', type=int,
                    help='Determine the time stamps from each tsteps' )


    # OPTIMISER
    parser.add_argument('--initialise_optimiser', type=list,
                        help='init_optimiser')
    parser.add_argument('--optimizer_type', type=list,
                        help='optimiser learning rate')
    parser.add_argument('--learning_rate', type=list,
                        help='optimiser learning rate')
    
    parser.add_argument('--sheduler_type', type=list,
                        help='type of sheduler')
    parser.add_argument('--sheduler_step', type= list,
                        help='sheduler_step') 
    parser.add_argument('--sheduler_gamma', type= list,
                        help='sheduler_factor')  

    parser.add_argument('--weight_decay', type=list,
                        help='optimiser weight decay')

    parser.add_argument('--new_training', type=bool,
                        help='new_training from ranfom intitialise')
    parser.add_argument('--load_experiment', type=str,
                        help='fine to load existing model ')   
    
    
    # TRAINING
    parser.add_argument('--training_protocol_type', type=str,
                        help='type of training strategy') 
    parser.add_argument('--training_protocols', type=list,
                        help='number of training protocols') 
    parser.add_argument('--number_of_training_protocol', type=int,
                        help='number of training protocols')    
    parser.add_argument('--epochs', type=list,
                        help='number of training epochs')
    parser.add_argument('--next_input_time_stamps', type=int,
                        help='number of output time stamps to pass for the next prediction (max: output_time_stamps )')
    parser.add_argument('--model_initialise_type', type=str,
                        help='type of initialise')
    parser.add_argument('--time_sampling_prediction', type=str,
                        help='time_samples for different samples within a batch: option: cosntant or random')        
    
    ## Loss 
    parser.add_argument('--loss_train_type', type=str,
                        help='type of training loss function')
    parser.add_argument('--loss_test_type', type=str,
                        help='type of test loss function')

    ## Dynamic weighting
    parser.add_argument('--dynamic_loss_weight_per_fpass', type=list,
                        help='loss weighting for each forward pass through the model: True or False for each training protocol')
    parser.add_argument('--dynamic_loss_weight_per_fpass_constant_parameter', type=list,
                    help='fpass constant parameter values')
    parser.add_argument('--dynamic_loss_weight_per_fpass_reversed', type=list,
                        help='loss weighting for each forward pass through the model: True or False for each training protocol')    
    parser.add_argument('--dynamic_loss_weight_per_tstamp', type=list,
                        help='loss weighting for each tstamps : True or False for each training protocol')  
    parser.add_argument('--dynamic_loss_weight_per_tstamp_constant_parameter', type=list,
                        help='tstamp_constant parameter values')  


    # SAVE
    parser.add_argument('--epoch_save_interval', type=str,
                        help='epoch_save_interval')
    parser.add_argument('--pretrained_model', type=str,
                        help='epoch_save_interval')
    parser.add_argument('--current_dir_path', type=str,
                        help='location of current working directory') 
     
     
    parser.add_argument('--result_save_path', type=str,
                        help='location for saving results') 
    parser.add_argument('--current_result_save_path', type=str,
                        help='location for saving current results') 
    parser.add_argument('--current_date_save_path', type=str,
                        help='location for saving current date') 


    # CURRICULUM LEARNING
    parser.add_argument('--training_type', type=str,
                        help='long or short horizon with shifting')
    parser.add_argument('--horizon', type=str,
                        help='number ofr autoregression for short horizon short horizon with shifting')
    parser.add_argument('--horizon_grad', type=str,
                        help='how many gradient are passed from horizons- MAX: number of Horizon')




    parser.set_defaults(
        mode = "train",

        test_only_path = " ",
        test_only_protocol_no = 1,
        test_only_epoch_index = -1,

        dataset_name = "E1",
        experiment = "testcase_2",
        dataset_train_path = " ",
        dataset_test_path = " ",

        n_train = 1,
        n_test = 1,
        batch_size_train = 1,
        batch_size_test = 1,
        total_t_range =   1,

        root_dir = None,
        seed = 250,
        model_type = "fno_1d",
        fno_hidden_dim = 16,
        fno_hidden_layers = 4,
        fno_modes = 16,
        input_shape_type = "concat",
    
        input_time_stamps = 10,
        input_tspace_range = 50,
        timestamps = [],
        

        output_time_stamps = 10,
        output_tspace_low = 50,
        output_tsteps_sampler_type = 1,
        output_tstamps_sampler_type = 1,


        model_mode = "variable_dt",
        model_input_operation = "concat",

        training_protocol_type = "dynamic_weighting",
        training_type = "short_horizon",
        horizon = 3,
        horizon_grad = 1,
        time_sampling_prediction = "random",


        training_protocols = [
            {"epochs": 100, 
             "no_input_tspace": 1,
             "dynamic_loss_weight_per_fpass": True,
             "dynamic_loss_weight_per_fpass_constant_parameter": 0.5,
             "dynamic_loss_weight_per_tstamp": True,
             "dynamic_loss_weight_per_tstamp_constant_parameter": 0.5,
             "dynamic_loss_weight_per_fpass_reversed": False,
             "initialise_optimiser": True,
             "optimiser_type": "adam",
             "sheduler_type":"steplr",
             "learning_rate": 0.001,
             "sheduler_step": 25,
             "sheduler_gamma":0.5,
             "weight_decay": 1e-4,
             "input_sampler_type": [0,],
             "input_sampler_type_dt": [0,],
             "output_sampler_type": [0,]
             },

            ],

        no_input_tspace = [1,1,1,1],
        initialise_optimiser = [],
        number_of_training_protocol = 3,
        optimiser_type = ["adam", ],
        learning_rate = [0.001,],
        sheduler_type = ["steplr",],
        sheduler_step = [25,],
        sheduler_gamma = [0.5,],
        weight_decay = [1e-4,], 
        epochs = [2,2,2,2],

        dynamic_loss_weight_per_fpass = [True, True, False, False],
        dynamic_loss_weight_per_fpass_reversed = [False, False, False, False],
        dynamic_loss_weight_per_tstamp = [False, False, False, False],

        dynamic_loss_weight_per_fpass_constant_parameter = [0.50, 0.51, 0.52, 0.51 ],
        dynamic_loss_weight_per_tstamp_constant_parameter = [0.51, 0.50, 0.51, 0.50 ],

        input_sampler_type = [0,],
        input_sampler_type_dt = [0,],
        output_sampler_type = [0,],
        
        next_input_time_stamps = 10,
        loss_train_type = "l2",
        loss_test_type = "l2",
        epoch_save_interval = 1,
        pretrained_model = None,


        current_dir_path = os.getcwd(),
        result_save_path = "",
        current_date_save_path = "",
        current_result_save_path = "",

        new_training =  None,
        load_experiment = None,
        model_initialise_type = None

    )


    args = parser.parse_args()


    return args