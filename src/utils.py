# -*- coding: utf-8 -*-
import inspect

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"


def check_params(fns, params, exceptions=None):
    if not exceptions:
        exceptions = []

    legal_params = []
    for fn in fns:
        legal_params += inspect.getargspec(fn)[0]
    legal_params = set(legal_params)

    for params_name in params:
        if params_name not in legal_params:
            if params_name not in exceptions:
                raise ValueError(
                        '{} is not a legal parameter'.format(params_name))


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result