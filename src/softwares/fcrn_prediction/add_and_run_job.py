# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from methods.cnn.fcrn import predict
from softwares.parser import parser_dataset, parser_job, parser_preprocessing, parser_postprocessing, \
    check_params_dataset, check_params_job, check_params_preprocessing, \
    check_params_postprocessing, check_params_cnn, remove_list

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"


def prediction_parser(argv):
    parser = ArgumentParser(prog="ConvNet Cell Counter Prediction",
                            description="A Cytomine software for cell counting")
    parser = parser_dataset(parser)
    parser = parser_job(parser)
    parser = parser_preprocessing(parser)
    parser = parser_postprocessing(parser)

    params, other = parser.parse_known_args(argv)
    params = check_params_dataset(params)
    params = check_params_job(params)
    params = check_params_preprocessing(params)
    params = check_params_postprocessing(params)
    params = check_params_cnn(params)

    params = remove_list(params)
    return params


if __name__ == '__main__':
    import sys

    parameters = prediction_parser(sys.argv[1:])
    predict(parameters)
