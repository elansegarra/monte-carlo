''' Auxiliary functions to go with Monte Carlo simulations '''

import pickle
import copy

def save_dgp_params(dgp_params, filename, omit_keys = None):
    '''
        This function saves the data generating process parameters going with
        a given Monte Carlo Simulation

        :param dgp_params:
        :param str filename:
        :param list omit_keys
    '''
    # Make a deep copy of the object
    params = copy.deepcopy(dgp_params)

    if omit_keys is None:
        omit_keys = []

    # Check object type (can only handle lists and dictionaries right now)
    assert isinstance(dgp_params, dict) or isinstance(dgp_params, list), "Can only handle lists and dicts right now."
    if isinstance(dgp_params, list):
        raise NotImplementedError("Not yet implemented saving of list objects")

    # Iterate through each item and replace those that are callable (ie a function/class) or in omit_keys
    for key, item in params.items():
        if callable(item) or (key in omit_keys):
            # Remove this element (replace with a note)
            params[key] = "unpicklable"

    # Pickle the object
    with open(filename, "wb") as outfile:
        pickle.dump(params, outfile)
