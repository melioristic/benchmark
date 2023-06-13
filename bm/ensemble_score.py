import numpy as np
import xarray as xr
from scipy.stats import entropy
from typing import Dict

def get_entropy_of_distributions(distributions: xr.Dataset, rank_dim="rank")-> xr.Dataset:
    return xr.apply_ufunc(entropy, distributions, input_core_dims=[[rank_dim]], vectorize=True)