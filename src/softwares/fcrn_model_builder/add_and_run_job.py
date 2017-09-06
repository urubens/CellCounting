# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from methods.cnn.fcrn import train
from softwares.parser import parser_dataset, parser_job, parser_preprocessing, parser_postprocessing, parser_subwindows, \
    parser_cnn, parser_augmentation, check_params_dataset, check_params_job, check_params_preprocessing, \
    check_params_postprocessing, check_params_subwindows, check_params_cnn, check_params_augmentation, remove_list

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"


def model_builder_parser(argv):
    parser = ArgumentParser(prog="ConvNet Cell Counter Model Builder",
                            description="A Cytomine software for cell counting")
    parser = parser_dataset(parser)
    parser = parser_job(parser)
    parser = parser_preprocessing(parser)
    parser = parser_postprocessing(parser)
    parser = parser_subwindows(parser)
    parser = parser_cnn(parser)
    parser = parser_augmentation(parser)

    params, other = parser.parse_known_args(argv)
    params = check_params_dataset(params)
    params = check_params_job(params)
    params = check_params_preprocessing(params)
    params = check_params_postprocessing(params)
    params = check_params_subwindows(params)
    params = check_params_cnn(params)
    params = check_params_augmentation(params, default_augmentation=True)

    params = remove_list(params)
    params.sw_extr_mode = 'random'
    return params


if __name__ == '__main__':
    import sys

    parameters = model_builder_parser(sys.argv[1:])
    # parameters = model_builder_parser("--dataset BMGRAZ --cnn_architecture FCRN-A --sw_input_size 256 --pre_transformer density --cytomine_working_path /Users/ulysse/Documents/Programming/TFE/tmp/ --model_file /Users/ulysse/Documents/Programming/cellcounting/models/fcrna_bmgraz.hdf5 -vvv".split(' '))
    train(parameters)
