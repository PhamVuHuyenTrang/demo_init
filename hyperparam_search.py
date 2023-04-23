import os
import sys
from datetime import datetime
import time
import pickle
import abc
import multiprocessing
from multiprocessing import Pool
from itertools import product
from pprint import pprint

def print_log(*args):
    print("[{}]".format(datetime.now()), *args)
    sys.stdout.flush()

def train_and_eval(param, save_path):
    """Accept a specific parameter setting, and run the training and evaluation
    procedure. 

    The training and evaluation procedure are specified by users.
    
    The evaluation procedure should return a `scalar` value to indicate the 
    performance of this parameter setting.

    Args:
    param -- A dict object specifying the parameters.
    save_path -- A string object of intermediate result saving path.

    Returns:
    A scalar value of evaluation.
    """
    pass
class HyperparameterSearch:
    """Singleprocessing hyperparameter search
    """
    def __init__(self, first_gpu=0, num_gpu=1,
                 save_dir=""):
        """
        Args:
        first_gpu -- The id of the first gpu in a consequtive sequence of gpus 
            in use.
        num_gpu -- The number of a consequtive sequence of gpus 
            in use. Specify 0 to indicate not using gpus.
        save_dir -- Save root path.
        """
        self.num_gpu = num_gpu
        if self.num_gpu > 0:
            self.first_gpu = first_gpu
        dt_now_str = datetime.now().strftime("d%Y%m%dt%H%M%S")
        if save_dir:
            self.save_dir = save_dir + '/' + "HyperSearch" + dt_now_str + '/'
        else:
            self.save_dir = "./" + "HyperSearch" + dt_now_str + '/'
        os.mkdir(self.save_dir)
        print_log(
            f"{self.num_gpu} GPUs in use. "
            f"Logs will be saved to {self.save_dir}.")

    def hyperparameter_optimization(self, train_and_eval, params):
        """Perform hyperparameter searching in the space defined by `params`. 
        
        Args:
        train_and_eval -- A function taking a specific parameter setting, and 
            run the training and evaluation procedure. 

            The training and evaluation procedure are specified by users.
            
            The evaluation procedure should return a `scalar` value to indicate 
            the performance of this parameter setting.

            Args:
            param -- A dict object specifying the parameters.
            save_path -- A string object of intermediate result saving path.

            Returns:
            A scalar value of evaluation.

        params -- A dict object defining the parameter space to search the
            optimal value. Each item must be iteratable and has the format:
                {
                    "param_1": [N, ...], 
                    "param_2": [N, ...],
                    ...
                }

        Returns:
        A list of dict object that contains the best configuration (note that 
            a repetition of the same values might happen);
        A list of dict object that contains the parameter setting;
        A list of tuples (evluation result, elapsed time) that corresponds to
        the parameter setting.
        """
        self.train_and_eval = train_and_eval # add for later usage

        pid = os.getpid()
        print(f"Ancestor process: {pid}")
        print("Parameter settings to be searched:")
        pprint(params, width=40)
        print_log("Search starts.")
        possible_combs = product(*params.values())
        # this format is required for multiprocessing
        params_list = [
            dict(zip(params.keys(), c))
                for c in possible_combs]

        eval_res_list = [] 
        elapsed_time_list = [] 
        for param in params_list:
            if hasattr(self, "train_and_eval"):
                start_time = time.time()
                eval_res = self.train_and_eval(param, save_path = self.save_dir)
                end_time = time.time()
                eval_res_list.append(eval_res)
                elapsed_time_list.append(end_time-start_time)
        best_eval_res = max(eval_res_list)
        best_args = [
            i for (i, res) in enumerate(eval_res_list) 
                if best_eval_res == res]
        best_param_list = [params_list[i] for i in best_args]
        # summary
        print_log(
            "Search finishes. "
            f"Best parameter setting with evaluation result {best_eval_res}:")
        pprint(best_param_list, width=40)
        print("\nOther settings")
        for p, r, t in zip(params_list, eval_res_list, elapsed_time_list):
            print('-'*78)
            pprint(p, width=80)
            print(f"Evaluation result: {r}")
            print(f"Elapsed time {t}")
        return best_param_list, params_list





