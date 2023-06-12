import numpy as np
import xarray as xr
from scipy.stats import entropy
from typing import Dict

def get_ranks(ground_truth:xr.Dataset, ensemble_data:xr.Dataset)->xr.Dataset:
    """Given an xarray Dataset containing ensemble predictions, and an xarray containing the ground truth, 
    return the rank of the ground_truth amongst the predicted ensemble members. 
    All inputs should have dimensions ["ensemble_member", "init_time"].

    Args:
        ground_truth (xr.Dataset): True data, of dimensions ["ensemble_member", "init_time"].
        ensemble_data (xr.Dataset): Predicted ensemble members, of dimensions ["ensemble_member", "init_time"].

    Returns:
        xr.Dataset: The ranks of the ground truth amongst the ensemble members, for each variable.
    """
    assert set(("ensemble_member", "init_time")) == set(ensemble_data.dims) == set(ground_truth.dims)

    ground_truth["ensemble_member"] = [-1]
    combined_ds = xr.merge((ground_truth, ensemble_data))
    return combined_ds.rank(dim="ensemble_member").sel({"ensemble_member":-1})


def get_ranks_distributions(ground_truth:xr.Dataset, ensemble_data:xr.Dataset)->Dict[str, np.ndarray]:
    """Given an xarray Dataset containing ensemble predictions, and an xarray containing the ground truth, 
    return the distribution of the ranks of the ground truth amongst the other ensemble members, 
    ie. count for each rank the ground_truth can be in, how often this rank is actually realized. 
    All inputs should have dimensions ["ensemble_member", "init_time"].

    Args:
        ground_truth (xr.Dataset): True data, of dimensions ["ensemble_member", "init_time"].
        ensemble_data (xr.Dataset): Predicted ensemble members, of dimensions ["ensemble_member", "init_time"]

    Returns:
        Dict[str, np.ndarray]: return how often the ground truth is at a given rank in [0, n_ensemble_members] for all variables and possible ranks.
    """
    ranks = get_ranks(ground_truth=ground_truth, ensemble_data=ensemble_data)

    n_ranks = len(ensemble_data.ensemble_member) + 1
    res = {}

    for var in ranks.data_vars:
        unique_values, counts = np.unique(ranks[var].values, return_counts=True)
        indices = np.array([np.where(unique_values==k)[0] for k in np.arange(1, n_ranks+1)])
        res[var] = counts[indices]
    return res

def get_entropy_of_distributions(distributions: Dict[str, np.ndarray])-> Dict[str, float]:
    """For a dict of multiple variables, compute the entropy of the corresponding rank distribution.

    Args:
        distributions (Dict[str, np.ndarray]): The variable names and the corresponding distributions 
        (datatype has to be int, ie. we work on count level).

    Returns:
        Dict[str, float]: The computed entropy for each variable.
    """
    for key, value in distributions.items():
        assert value.dtype == "int64", "For variable {} the ranks are not of type int64".format(key)
    

    res = {}
    for key, value in distributions.items():
        res[key] = entropy(distributions[key])
    return res