import pandas as pd
import numpy as np
from tqdm import tqdm
import inspect  # This allows us to check the arguments of a function
import warnings


def run_monte_carlo(dgp_params, sample_fun, est_fun, num_sims,
                        const_aux_params = None, mc_seed = None,
                        disable_progress = False):
    '''
        Performs the specified monte carlo simulations (draw a sample, estimate
        somethign on it, and repeat) and returns the results in a dataframe or
        dictionary.

        :param dict dgp_params: Contains the data generating process parametrization
            that will be passed to sample_fun
        :param function sample_fun: Function which generates a sample
            Should have 2 arguments: dgp_dict and const_param_dict (and an optional arg called 'seed')
        :param function est_fun: Function which applies a transformation to sample
            Should have 2 arguments: sample and const_param_dict
        :param int num_sims: The number of simulations to conduct
        :param dict const_aux_params: contains variables that are constant across
            estimations (this gets passed to est_fun)
        :param int/list mc_seed: sets a seed for all MC simulations (default is
            random) or can be a list of seeds to use for the sims
        :param bool disable_progress: Indicates whether to disable the progress bar (tqdm)
    '''

    # Draw the seeds for each simulation
    if (mc_seed is not None) and (isinstance(mc_seed,int)):
        np.random.seed(mc_seed)
        seed_list = np.random.randint(1, high = num_sims*1000, size=num_sims)
    if (mc_seed is not None) and (isinstance(mc_seed,list)):
        assert len(mc_seed) == num_sims, "mc_seed needs to have a length equal to num_sims."
        seed_list = mc_seed
    else:
        seed_list = np.random.randint(1, high = num_sims*1000, size=num_sims)

    # Checking the arguments of the passed functions
    s_args = inspect.getargspec(sample_fun)
    e_args = inspect.getargspec(est_fun)

    # Iterate over each simulation
    # TODO: Implement parallelization
    mc_res_list = [None]*num_sims
    for i in tqdm(range(num_sims), disable=disable_progress):
        # Draw the sample (with the appropriate seed)
        if 'seed' in inspect.getargspec(sample_fun).args:
            mc_sample = sample_fun(dgp_params, const_aux_params, seed = seed_list[i])
        else:
            dgp_params['seed'] = seed_list[i]
            mc_sample = sample_fun(dgp_params, const_aux_params)

        # Apply the function (eg. estimate)
        fun_result = est_fun(mc_sample, const_aux_params)

        # Save the result (depending on its type)
        if isinstance(fun_result, list):
            fun_result = [i, seed_list[i]] + fun_result
        elif isinstance(fun_result, dict):
            if "sim" not in fun_result:  fun_result['sim'] = i
            if "seed" not in fun_result: fun_result['seed'] = seed_list[i]
        else:
            warnings.warn("The return of est_fun is not an iterable. Treating as a singleton")
            # raise TypeException("The return value of 'est_fun' is not recognized. Should be either a list or a dictionary.")
            fun_result = [i, seed_list[i], fun_result]
        mc_res_list[i] = fun_result

    # Assemble the results into a dataframe
    # if isinstance(mc_res_list[0], list):
    #     mc_result = pd.DataFrame(data=mc_res_list, columns = ['sim', 'seed']+['res']*(len(mc_res_list[0])-2))
    if isinstance(mc_res_list[0], dict):
        # Reordering the columns so sim and seed are first
        other_cols = list(mc_res_list[0].keys() - {'sim', 'seed'})
        mc_result = pd.DataFrame(mc_res_list)[['sim', 'seed']+other_cols].copy()
    else:
        # TODO: Find way to pass names of columns
        # Giving placeholder names for all result columns
        res_cols = ['res_'+str(i) for i in range(1,len(mc_res_list[0])-1)]
        mc_result = pd.DataFrame(data=mc_res_list, columns = ['sim', 'seed']+res_cols)

    return mc_result
