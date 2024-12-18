import numpy as np
import networkx as nx
import pandas as pd
import pickle
import dill
import os
import pdb

#================================================================
# General functions for saving/loading data
#----------------------------------------------------------------
class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    Note: doesn't work for complex nested dictionaries
    adapted from: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)
    
def serialize_model(obj, path):
    with open(path, 'wb') as fh:
        return dill.dump(obj, fh)

def deserialize_model(path):
    with open(path, 'rb') as fh:
        return dill.load(fh)