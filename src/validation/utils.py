# -*- coding: utf-8 -*-
import six
import copy
from scipy import sparse

import numpy as np
from sklearn.base import _first_and_last_element
from sklearn.model_selection import KFold, GroupKFold, LeavePGroupsOut
from sldc import Image, TileBuilder, DefaultTileBuilder, TileTopologyIterator

from data.io import open_image


__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"


def mk_tt_split(X, y, labels, test_labels):
    """
    Perform a train/test split based on labels.
    
    Parameters
    ----------
    X : array_like
        Input samples
    y : array_like
        Output samples
    labels : array_like
        Set of labels
    test_labels : array_like
        Set of test labels, that is, a subset of `labels`.

    Returns
    -------
    X_LS
    y_LS
    labels_LS
    X_TS
    y_TS
    labels_TS
    
    """
    test_set_labels = np.unique(test_labels)
    ts = np.in1d(labels, test_set_labels)
    ls = np.logical_not(ts)
    return (np.asarray(X[ls]), np.asarray(y[ls]), np.asarray(labels[ls]),
            np.asarray(X[ts]), np.asarray(y[ts]), np.asarray(labels[ts]))


def cv_strategy(parameters):
    if parameters.cv_mode == 'GKF':
        return GroupKFold(n_splits=parameters.cv_param)
    elif parameters.cv_mode == 'LPGO':
        return LeavePGroupsOut(n_groups=parameters.cv_param)
    else:
        raise ValueError("Unknown CV mode")


def mk_param_grid(param_dict, param_keys):
    ret = param_dict.copy()
    for k in param_dict.keys():
        if k not in param_keys:
            del ret[k]
    return ret


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def clone(estimator, safe=True):
    """Constructs a new estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It yields a new estimator
    with the same parameters that has not been fit on any data.

    Parameters
    ----------
    estimator: estimator object, or list, tuple or set of objects
        The estimator or group of estimators to be cloned

    safe: boolean, optional
        If safe is false, clone will fall back to a deepcopy on objects
        that are not estimators.

    """
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params'):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a scikit-learn estimator "
                            "as it does not implement a 'get_params' methods."
                            % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in six.iteritems(new_object_params):
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is param2:
            # this should always happen
            continue
        if isinstance(param1, np.ndarray):
            # For most ndarrays, we do not test for complete equality
            if not isinstance(param2, type(param1)):
                equality_test = False
            elif (param1.ndim > 0
                    and param1.shape[0] > 0
                    and isinstance(param2, np.ndarray)
                    and param2.ndim > 0
                    and param2.shape[0] > 0):
                equality_test = (
                    param1.shape == param2.shape
                    and param1.dtype == param2.dtype
                    and (_first_and_last_element(param1) ==
                         _first_and_last_element(param2))
                )
            else:
                equality_test = np.all(param1 == param2)
        elif sparse.issparse(param1):
            # For sparse matrices equality doesn't work
            if not sparse.issparse(param2):
                equality_test = False
            elif param1.size == 0 or param2.size == 0:
                equality_test = (
                    param1.__class__ == param2.__class__
                    and param1.size == 0
                    and param2.size == 0
                )
            else:
                equality_test = (
                    param1.__class__ == param2.__class__
                    and (_first_and_last_element(param1) ==
                         _first_and_last_element(param2))
                    and param1.nnz == param2.nnz
                    and param1.shape == param2.shape
                )
        else:
            # fall back on standard equality
            equality_test = param1 == param2
        if equality_test:
            pass
        else:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'does not seem to set parameter %s' %
                               (estimator, name))

    return new_object


#### Tiles

class MyImage(Image):
    def __init__(self, np_array):
        self.np_array = np.array(np_array)

    @property
    def width(self):
        return self.np_array.shape[0]

    @property
    def height(self):
        return self.np_array.shape[1]

    @property
    def np_image(self):
        return self.np_array

    @property
    def channels(self):
        return self.np_array.shape[2]

    def tile_iterator(self, builder, max_width=1024, max_height=1024, overlap=0):
        """Build and return a tile iterator that iterates over the image

        Parameters
        ----------
        builder: TileBuilder
            The builder to user for actually constructing the tiles while iterating over the image
        max_width: int (optional, default: 1024)
            The maximum width of the tiles to build
        max_height: int (optional, default: 1024)
            The maximum height of the tiles to build
        overlap: int (optional, default: 0)
            The overlapping between tiles

        Returns
        -------
        iterator: TileTopologyIterator
            An iterator that iterates over a tile topology of the image
        """
        topology = self.tile_topology(builder, max_width=max_width, max_height=max_height, overlap=overlap)
        return TileTopologyIterator(builder, topology)

if __name__ == '__main__':
    pass
