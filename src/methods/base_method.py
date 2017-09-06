# -*- coding: utf-8 -*-
import copy
import inspect
import numpy as np

from features.postprocessing import non_maximum_suppression
from jobs.logger import StandardOutputLogger, Logger

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"

class BaseMethod(object):
    def __init__(self, build_fn=None, logger=StandardOutputLogger(Logger.INFO), **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params
        self.logger = logger

    def get_params(self, **params):
        """Gets parameters for this estimator.

        # Arguments
            **params: ignored (exists for API compatiblity).

        # Returns
            Dictionary of parameter names mapped to their values.
        """
        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn': self.build_fn})
        return res

    def set_params(self, **params):
        """Sets the parameters of this estimator.

        # Arguments
            **params: Dictionary of parameter names mapped to their values.

        # Returns
            self
        """
        self.sk_params.update(params)
        return self

    def filter_sk_params(self, fn, override=None, exceptions=[]):
        """Filters `sk_params` and return those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override sk_params

        # Returns
            res : dictionary dictionary containing variables
                in both sk_params and fn's arguments.
        """
        override = override or {}
        res = {}
        fn_args = inspect.getargspec(fn)[0]
        for name, value in self.sk_params.items():
            if name in fn_args and name not in exceptions:
                res.update({name: value})
        res.update(override)
        return res

    def postprocessing(self, X, **post_params):
        return np.squeeze([non_maximum_suppression(x, **post_params) for x in X])