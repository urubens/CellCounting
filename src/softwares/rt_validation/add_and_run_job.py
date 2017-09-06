# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from methods.rt.rt import train
from softwares.parser import parser_dataset, parser_job, parser_preprocessing, parser_postprocessing, parser_subwindows, \
    parser_cnn, parser_augmentation, check_params_dataset, check_params_job, check_params_preprocessing, \
    check_params_postprocessing, check_params_subwindows, check_params_cnn, check_params_augmentation, parser_cv, \
    check_params_cv, parser_randomizedtrees, check_params_randomizedtrees

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"


def validation_parser(argv):
    parser = ArgumentParser(prog="Randomized trees Cell Counter Validation",
                            description="A Cytomine software for cell counting")
    parser = parser_dataset(parser)
    parser = parser_job(parser)
    parser = parser_preprocessing(parser)
    parser = parser_postprocessing(parser)
    parser = parser_subwindows(parser)
    parser = parser_randomizedtrees(parser)
    parser = parser_augmentation(parser)
    parser = parser_cv(parser)

    params, other = parser.parse_known_args(argv)
    params = check_params_dataset(params)
    params = check_params_job(params)
    params = check_params_preprocessing(params)
    params = check_params_postprocessing(params)
    params = check_params_subwindows(params)
    params = check_params_randomizedtrees(params)
    params = check_params_augmentation(params, default_augmentation=False)
    params = check_params_cv(params)


    return params


if __name__ == '__main__':
    import sys

    parameters = validation_parser(sys.argv[1:])
    # parameters = validation_parser("--cv_mode GKF --dataset BMGRAZ --sw_extr_mode random --sw_extr_npi 10 --forest_method ET-regr --forest_min_samples_split 200 --forest_n_estimators 2 --pre_transformer density --cytomine_working_path /Users/ulysse/Documents/Programming/TFE/tmp/ --model_file /Users/ulysse/Documents/Programming/cellcounting/models/t.hdf5 --cv_save_to /Users/ulysse/Documents/Programming/cellcounting/results/t.csv -vvv".split(' '))

    train(parameters, mk_cv=True)
