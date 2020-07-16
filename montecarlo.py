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
        something on it, and repeat) and returns the results in a dataframe.

        :param dict dgp_params: Contains the data generating process parametrization
            that will be passed to sample_fun
        :param function sample_fun: Function which generates a sample
            Req 2 arguments: dgp_dict and const_param_dict (and an optional arg called 'seed')
            Returns an object that is parseable by 'est_fun'
        :param function est_fun: Function which applies a transformation to sample
            Should have 2 arguments: sample and const_param_dict
            Returns an iterable (i.e. dict or list of estimation results)
        :param int num_sims: The number of simulations to conduct
        :param dict const_aux_params: contains variables that are constant across
            estimations (this gets passed to est_fun)
        :param int/list mc_seed: sets a seed for all MC simulations (default is
            random) or can be a list of seeds to use for the sims
        :param bool disable_progress: Indicates whether to disable the progress bar (tqdm)

        :return: pandas.DataFrame where each row is a single simulation result
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

        # Apply the function (eg. estimate things)
        fun_result = est_fun(mc_sample, const_aux_params)

        # Save the result (depending on its type)
        if isinstance(fun_result, list):
            fun_result = [i, seed_list[i]] + fun_result
        elif isinstance(fun_result, dict):
            if "sim" not in fun_result:  fun_result['sim'] = i
            if "seed" not in fun_result: fun_result['seed'] = seed_list[i]
        else:
            warnings.warn("The return of est_fun is not an iterable. Treating as a singleton.")
            fun_result = [i, seed_list[i], fun_result]
        mc_res_list[i] = fun_result

    # Assemble the results into a dataframe
    # if isinstance(mc_res_list[0], list):
    #     mc_result = pd.DataFrame(data=mc_res_list, columns = ['sim', 'seed']+['res']*(len(mc_res_list[0])-2))
    if isinstance(mc_res_list[0], dict):
        # Reordering the columns so sim and seed are first
        other_cols = [col for col in mc_res_list[0].keys() if col not in ['sim', 'seed']]
        mc_result = pd.DataFrame(mc_res_list)[['sim', 'seed']+other_cols].copy()
    else:
        # TODO: Find way to pass names of columns
        # Giving placeholder names for all result columns
        res_cols = ['res_'+str(i) for i in range(1,len(mc_res_list[0])-1)]
        mc_result = pd.DataFrame(data=mc_res_list, columns = ['sim', 'seed']+res_cols)

    return mc_result

def stability_check(mc_df, use_col, sample_pcts = [0.9, 0.75, 0.5], num_samples = 3,
                    seed = None, plot_running_mean = False):
    '''
        Checks the stability of the current set of simulations to help judge whether
        more simulations need to be run.

        :param pd.DataFrame mc_df: contains the MC results (output from run_monte_carlo function)
        :param str use_col: which column should be used for test
        :param list sample_pcts: list of floats indicating the sample sizes (in percentage) to be drawn
        :param num_samples int: Number of samples in each percent to draw
        :param seed int: For replication of stability analysis
    '''

    # Check that the use_col is in the df
    if use_col not in mc_df.columns:
        print(f"Column, {use_col}, not found in the passed MC simulation dataframe.")

    # Calculating the mean and std deviation for entire simulation study
    mean_100_pct = mc_df[use_col].mean()
    sd_100_pct = mc_df[use_col].std()
    total_sims = mc_df.shape[0]
    print(f"100% of ({total_sims}) Simulations:")
    print(f"Mean('{use_col}'): {mean_100_pct}")
    print(f"SDev('{use_col}'): {sd_100_pct}")

    # Setting seed if passed
    np.random.seed(seed)

    # Calculating means of specified sample sizes
    stability_res = dict()
    col_names = []
    for sample_pct in sample_pcts:
        col_name = str(round(100*sample_pct))+'pct'
        sample_size = int(np.round(total_sims*sample_pct))
        all_samples = [sample_size]
        for i in range(num_samples):
            all_samples.append(mc_df[use_col].sample(sample_size).mean())
        stability_res[col_name] = all_samples

    # Assembling means into a dataframe (with appropriate column names)
    stability_df = pd.DataFrame(stability_res).T
    new_cols = {0:'Size'}
    new_cols.update({col:'Sample_'+str(col) for col in stability_df.columns if col != 0})
    stability_df.rename(columns = new_cols, inplace=True)
    stability_df['Size'] = stability_df['Size'].astype(int)

    if plot_running_mean:
        x = 3

    return  stability_df
def outlier_diagnostic(mc_df, cols_to_check, IQR_thresh = 1.5, flag_rows = False,
                    verbose = False):
    '''
        Checks if there are outliers in the estimates across several columns
        and prints a short summary of the diagnostic results.

        :param pd.DataFrame mc_df: Contains the MC results (returned by run_mc)
        :param list cols_to_check: Columns in mc_def to check
        :param float IQR_thresh: Threshhold used to identify an outlier. Outliers
            are any rows falling outside of (Q1 - IQR_thresh*IQR, Q3 + IQR_thresh*IQR)
        :param bool/str flag_rows: Indicates whether to add a column which flags
            outlier rows (if a string is passed, then that will be the column name)
    '''
    # Check that all columns exist in the passed df
    cols_not_found = set(cols_to_check) - set(mc_df.columns)
    if len(cols_not_found) > 0:
        warnings.warn(f"Some columns ({cols_not_found}) were not found in the df so they are ignored.")
        cols_to_check = list(set(cols_to_check) - cols_not_found)

    # Iterate over each column and check for outliers (and collecting along the way)
    all_outliers = set()
    low_outlier_cts, high_outlier_cts = [], []
    for col in cols_to_check:
        # Calculating first quartile, third quartile, and IQR
        Q1, Q3 = mc_df[col].quantile(0.25), mc_df[col].quantile(0.75)
        IQR = Q3 - Q1
        low_outliers = set(mc_df[mc_df[col] < Q1 - IQR_thresh*IQR].index)
        low_outlier_cts.append(len(low_outliers))
        high_outliers = set(mc_df[mc_df[col] > Q3 + IQR_thresh*IQR].index)
        high_outlier_cts.append(len(high_outliers))
        # Updating list of all outliers
        all_outliers = all_outliers | low_outliers
        all_outliers = all_outliers | high_outliers

    # Printing out verbose results if specified
    if verbose:
        outlier_res = pd.DataFrame({'Low_Outliers': low_outlier_cts,
                                    'High_Outliers': high_outlier_cts},
                                    index = cols_to_check)
        print(outlier_res)

    # Printing out very short summary
    if len(all_outliers) > 0:
        print(f"A total of {len(all_outliers)} outlier(s) were found across {cols_to_check}.")
    else:
        print(f"No outliers found across {cols_to_check}.")

    # Add a column that flags entires (if specified)
    # TODO: Make flag include which cols were found
    if isinstance(flag_rows, bool) and flag_rows:
        mc_df['outlier'] = False
        mc_df.loc[all_outliers, 'outlier'] = True
    elif isinstance(flag_rows, str):
        mc_df[flag_rows] = False
        mc_df.loc[all_outliers, flag_rows] = True
