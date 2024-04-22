import pdb
import pickle
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import h5py
from constant_autoregression.util import Printer
from constant_autoregression.dataset.mppde1d import CE, HDF5Dataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = Printer(n_digits=6)


def load_data(args, **kwargs):

    # train_loader = None
    # valid_loader = None
    # test_loader = None



    if args.dataset_name.endswith("E1"):
        args.total_t_range = 250
        uniform_sample = -1
        super_resolution = [args.total_t_range,200]
        base_resolution = [args.total_t_range,100]
        n_workers = 4
        pde = CE(device=device)

        args.time_stamps = [i*0.004 for i in range(0,args.total_t_range)]

        train_string = f'dataset/data/{pde}_train_E1.h5'
        valid_string = f'dataset/data/{pde}_valid_E1.h5'
        test_string = f'dataset/data/{pde}_test_E1.h5'

        p.print(f"Load dataset: {train_string}")
        train_dataset = HDF5Dataset(train_string, pde=pde, mode='train', base_resolution=base_resolution, super_resolution=super_resolution, uniform_sample=uniform_sample)
        #import pdb; pdb.set_trace()
        #train_dataset[f'pde_{super_resolution[0]}-{super_resolution[1]}'][:args.n_train]
        
        p.print(f"Load dataset: {valid_string}")
        valid_dataset = HDF5Dataset(valid_string, pde=pde, mode='valid', base_resolution=base_resolution, super_resolution=super_resolution, uniform_sample=uniform_sample)
        #valid_dataset[f'pde_{super_resolution[0]}-{super_resolution[1]}'][:args.n_test]

        p.print(f"Load dataset: {test_string}")
        test_dataset = HDF5Dataset(test_string, pde=pde, mode='test', base_resolution=base_resolution, super_resolution=super_resolution, uniform_sample=uniform_sample)
        #test_dataset[f'pde_{super_resolution[0]}-{super_resolution[1]}'][:args.n_test]

    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size_train,
                                shuffle=True,
                                num_workers= n_workers,
                                )

    valid_loader = DataLoader(valid_dataset,
                            batch_size=args.batch_size_test,
                            shuffle=False,
                            num_workers=n_workers,
                            )

    test_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size_test,
                                shuffle=False,
                                num_workers=n_workers,
                                )
    
    #import pdb; pdb.set_trace()
    return train_loader, valid_loader, test_loader

    


class Input_Batch():
    """Object for holding a batch of data with mask during training."""

    def __init__(self, data, input_sample_type, input_range, total_range, no_of_input, dt_input = None ):
        if input_sample_type == 1:
            self.input_indicies = self.input_indicies_1(input_range, no_of_input)
        elif input_sample_type == 2:
            self.input_indicies = self.input_indicies_2(input_range, no_of_input, dt_input)
        elif input_sample_type == 3:
            assert dt_input != None
            self.input_indicies = self.input_indicies_3(input_range, no_of_input, dt_input)
        else:
            raise TypeError("Specify input_sample_type: 1 (non_independent sampes ) OR 2 (Independent samples )")
        self.input = data[..., self.input_indicies]

    @staticmethod
    def input_indicies_1(input_range, n):
        """
        generate n random input samples from the range (0, input_range )
        """
        return torch.sort(torch.randint(input_range, (n,)))[0]

    @staticmethod
    def input_indicies_2(input_range, n, dt_input ):
        """
        generate n constant dt input independent samples from the range (0, input_range )
        """
        assert dt_input < 6
        return torch.arange(0,input_range,1)[::dt_input][:n]




class Output_Batch():
    """Object for holding a batch of data with mask during training."""

    def __init__(self,input_indicies=None, data=None, output_sample_type=None, total_range=None, no_of_output=None ):
        if output_sample_type == 1:
            self.output_indicies = self.output_indicies_1(input_indicies, no_of_output, total_range)
        elif output_sample_type == 2:
            self.output_indicies = self.output_indicies_2(input_indicies, no_of_output, total_range)
        else:
            raise TypeError("Specify input_sample_type: 1 (non_independent sampes ) OR 2 (Independent samples )")
        self.output = data[..., self.output_indicies]

    @staticmethod
    def output_indicies_1(input_indicies, no_of_output,total_range):
        """
        generate n variable dt output samples from the remining indicies
        """
        output_indicies = torch.sort(torch.randint(total_range-input_indicies[-1]-1, (no_of_output,)))[0]
        return output_indicies + input_indicies[-1]+ 1


    @staticmethod
    def output_indicies_2(input_indicies, no_of_output,total_range):
        """
        generate n constant dt output samples from the remining indicies
        """
        last_ind_id = input_indicies[-1]
        dt = int(1)
        return torch.arange(last_ind_id+dt, total_range)[::dt] #output_indicies  #+ input_indicies[-1] + 1



class no_of_output_space():
    """Generate a space of number of output predictions

    args: out_low = smallest number of output predictions
          out_low = highest number of output predictions

    return:
      output_space: space of number of output predictions
      output_tray: number of samples from output space
    """

    def __init__(self,out_low=None, out_high=None, output_space_type=None, number_of_samp_per_batch=None, t_pred_steps=None, predefined=None):
        if output_space_type == 1:
            self.output_tray = self.output_space_1(out_low, out_high, t_pred_steps)
        elif output_space_type == 2:
            self.output_tray = self.output_space_2(out_low, out_high, t_pred_steps)
        elif output_space_type == 3:
            self.output_tray = predefined
        else:
            raise TypeError("Specify output_space_type: 1 ")


    @staticmethod
    def output_space_1(out_low,out_high, n = 25):
        """
        generate n output space
        """
        if n == 1:
            return torch.arange(out_low,out_high+1,10)
        else:
            return torch.arange(out_low,out_high+1,n)

    @staticmethod
    def output_space_2(out_low,out_high, n = 25):
        """
        generate n output space
        """
        if n == 1:
            return torch.arange(out_low,out_high+1,10)
        else:
            return torch.arange(out_low,out_high+1,n)



def load_dataset_E1(args):

        hdf5_train_file = h5py.File(args.dataset_train_path, 'r')
        hdf5_test_file = h5py.File(args.dataset_test_path, 'r')
        hdf5_valid_file = h5py.File(args.dataset_valid_path, 'r')

        train_loaded_data = hdf5_train_file['train']['pde_250-100'][:]
        test_loaded_data = hdf5_test_file['test']['pde_250-100'][:]
        valid_loaded_data = hdf5_valid_file['valid']['pde_250-100'][:]

        train_tensor =  train_loaded_data.squeeze()
        train_data = torch.from_numpy(train_tensor).float()

        test_tensor =  test_loaded_data.squeeze()
        test_data = torch.from_numpy(test_tensor).float()

        valid_tensor =  valid_loaded_data.squeeze()
        valid_data = torch.from_numpy(valid_tensor).float()


        x_train = train_data[:args.n_train,...]
        x_test = test_data[:args.n_test,...]
        x_valid = valid_data[:args.n_test,...]

        args.total_t_range = 250
        args.time_stamps = [i*0.004 for i in range(0,args.total_t_range)]

        res = x_train.shape[1]
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,x_train, x_train, x_train), batch_size=args.batch_size_train, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, x_test, x_test), batch_size=args.batch_size_test, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, x_valid, x_valid, x_valid), batch_size=args.batch_size_test, shuffle=False)

        #data = {"train_loader":train_loader, "test_loader": test_loader, "timestamps":timestamps}
        #import pdb; pdb.set_trace()
        return train_loader,valid_loader, test_loader