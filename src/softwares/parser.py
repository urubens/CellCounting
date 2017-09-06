# -*- coding: utf-8 -*-
import os
import tempfile

from data.cytomine_identifiers import DATASETS, CYTOMINE_KEYS, TEST_SET
from data.io import make_dirs

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"

TUNABLE_PARAMETERS = ['pre_alpha', 'sw_input_size', 'sw_output_size',
                       'sw_size', 'sw_extr_stride', 'sw_extr_score_thres', 'sw_extr_ratio',
                       'sw_extr_npi', 'forest_method', 'forest_n_estimators',
                       'forest_min_samples_split', 'forest_max_features', 'initializer',
                       'regularizer', 'batch_normalization', 'learning_rate', 'momentum', 'decay',
                       'epochs', 'batch_size', 'nesterov']  # 'post_threshold', 'post_sigma',


def check_default(param, default_value=None, return_list=True):
    if param is None or len(param) == 0:
        if return_list:
            return [default_value]
        else:
            return default_value

    return param


def check_max_features(max_features):
    ret = []
    for mf in max_features:
        if '.' in mf:
            ret.append(float(mf))
        elif mf == 'sqrt':
            ret.append(mf)
        else:
            ret.append(int(mf))

    return ret


def remove_list(params, param_names=None):
    if param_names is None:
        param_names = TUNABLE_PARAMETERS

    for param_name in param_names:
        v = getattr(params, param_name, None)
        if isinstance(v, list):
            setattr(params, param_name, v[0])

    return params


def parser_dataset(parser):
    # Option 1: Dataset name provided
    parser.add_argument('--dataset', dest='dataset', type=str,
                        help="Dataset name")

    # Option 2: Download from Cytomine
    parser.add_argument('--cytomine_host', dest='cytomine_host',
                        default='demo.cytomine.be', help="The Cytomine host")
    parser.add_argument('--cytomine_public_key', dest='cytomine_public_key',
                        help="The Cytomine public key")
    parser.add_argument('--cytomine_private_key', dest='cytomine_private_key',
                        help="The Cytomine private key")
    parser.add_argument('--cytomine_base_path', dest='cytomine_base_path',
                        default='/api/', help="The Cytomine base path")
    parser.add_argument('--cytomine_working_path', dest='cytomine_working_path',
                        default=None, help="The working directory (eg: /tmp)")
    parser.add_argument('--cytomine_software', dest='cytomine_software', type=int,
                        help="The Cytomine software identifier")
    parser.add_argument('--cytomine_project', dest='cytomine_project', type=int,
                        help="The Cytomine project identifier")
    parser.add_argument('--cytomine_force_download', dest='cytomine_force_download', type=bool, default=False,
                        help="Force download from Cytomine or not")

    parser.add_argument('--cytomine_object_term', dest='cytomine_object_term', type=int,
                        help="The Cytomine identifier of object term")
    parser.add_argument('--cytomine_object_user', dest='cytomine_object_user', type=int,
                        help="The Cytomine identifier of object owner")
    parser.add_argument('--cytomine_object_reviewed_only', dest='cytomine_object_reviewed_only', type=bool,
                        help="Weither objects have to be reviewed or not")

    parser.add_argument('--cytomine_roi_term', dest='cytomine_roi_term', type=int,
                        default=None,
                        help="The Cytomine identifier of region of interest term")
    parser.add_argument('--cytomine_roi_user', dest='cytomine_roi_user', type=int,
                        help="The Cytomine identifier of ROI owner")
    parser.add_argument('--cytomine_roi_reviewed_only', dest='cytomine_roi_reviewed_only', type=bool,
                        help="Weither ROIs have to be reviewed or not")

    parser.add_argument('--cv_labels', dest='cv_labels', type=str, default='image',
                        help="")
    parser.add_argument('--image_as_roi', dest='image_as_roi', type=bool, default=True,
                        help="The mean radius of object to detect")

    parser.add_argument('--mean_radius', dest='mean_radius', type=int,
                        help="The mean radius of object to detect")
    return parser


def check_params_dataset(params, default_working_path=None):
    if default_working_path is None:
        default_working_path = os.path.join(tempfile.gettempdir(), "cytomine")

    # If a dataset name is provided, some parameters are already hard-coded.
    if params.dataset is not None:
        if params.dataset not in DATASETS:
            raise ValueError('Unknown dataset: "{}"'.format(params.dataset))
        dataset = DATASETS[params.dataset]
        params.cytomine_host = dataset['host']
        params.cytomine_public_key = CYTOMINE_KEYS[dataset['host']]['public_key']
        params.cytomine_private_key = CYTOMINE_KEYS[dataset['host']]['private_key']
        params.cytomine_project = dataset['id']
        params.cytomine_object_term = dataset['cell_term']
        params.cytomine_object_user = dataset['users']
        params.cytomine_object_reviewed_only = dataset['reviewed_only']
        params.cytomine_roi_term = dataset['roi_term']
        params.cytomine_roi_user = dataset['users']
        params.cytomine_roi_reviewed_only = dataset['reviewed_only']
        params.cv_test_images = TEST_SET[params.dataset]
        params.cv_labels = dataset['labels']
        params.image_as_roi = dataset['image_as_roi']
        params.mean_radius = dataset['mean_radius']
        params.cv_epsilon = dataset['cv_epsilon']
        params.post_min_dist = dataset['post_min_dist']

    if params.cytomine_working_path is None:
        params.cytomine_working_path = default_working_path

    make_dirs(params.cytomine_working_path)
    return params


def parser_job(parser):
    parser.add_argument('--n_jobs', dest='n_jobs', type=int, default=1,
                        help="Number of jobs")
    parser.add_argument('--verbose', '-v', dest='verbose', default=0, action='count',
                        help="Level of verbosity (3=INFO, 4=DEBUG)")
    parser.add_argument('--model_file', dest='model_file', type=str,
                        default=None, help="Model path")
    return parser


def check_params_job(params):
    if params.model_file is None:
        params.model_file = os.path.join(params.cytomine_working_path, "models")
    make_dirs(params.model_file, remove_filename=True)
    return params


def parser_preprocessing(parser):
    parser.add_argument('--pre_transformer', dest='pre_transformer',
                        default=None, choices=['edt', 'euclidean_distance_transform',
                                               'density', 'proximity', None, 'None'],
                        help="Scoremap transformer (None, edt, euclidean_distance_transform, "
                             "density, proximity)")
    parser.add_argument('--pre_alpha', dest='pre_alpha',
                        action='append', type=int,
                        help="Exponential decrease rate of distance (if EDT)")
    return parser


def check_params_preprocessing(params):
    params.pre_transformer = check_default(params.pre_transformer, None, return_list=False)
    params.pre_alpha = check_default(params.pre_alpha, 5)
    return params


def parser_postprocessing(parser):
    parser.add_argument('--post_threshold', dest='post_threshold',
                        action='append', type=float,
                        help="Post-processing discarding threshold")
    parser.add_argument('--post_sigma', dest='post_sigma',
                        action='append', type=float,
                        help="Std-dev of Gauss filter applied to smooth prediction")
    parser.add_argument('--post_min_dist', dest='post_min_dist', type=int,
                        help="Minimum distance between two peaks")
    return parser


def check_params_postprocessing(params):
    params.post_threshold = check_default(params.post_threshold, 0.5)
    params.post_sigma = check_default(params.post_sigma, None)
    if params.dataset is None:
        params.post_min_dist = check_default(params.post_min_dist, 7, return_list=False)
    return params


def parser_subwindows(parser):
    parser.add_argument('--sw_input_size', dest='sw_input_size',
                        action='append', type=int,
                        help="Size of input subwindow")
    parser.add_argument('--sw_output_size', dest='sw_output_size',
                        action='append', type=int,
                        help="Size of output subwindow (ignored for FCRN)")

    parser.add_argument('--sw_extr_mode', dest='sw_extr_mode',
                        # action='append',
                        choices=['random', 'sliding', 'scoremap_constrained'],
                        help="Mode of extraction (random, sliding, scoremap_constrained)")
    parser.add_argument('--sw_extr_stride', dest='sw_extr_stride',
                        action='append', type=int,
                        help="Stride for subwindows extraction "
                             "(if 'sliding' mode)")
    parser.add_argument('--sw_extr_score_thres', dest='sw_extr_score_thres',
                        action='append', type=float,
                        help="Minimum threshold to be foreground in subwindows extraction"
                             "(if 'scoremap_constrained' mode)")
    parser.add_argument('--sw_extr_ratio', dest='sw_extr_ratio',
                        action='append', type=float,
                        help="Ratio of background subwindows extracted in subwindows "
                             "extraction (if 'scoremap_constrained' mode)")
    parser.add_argument('--sw_extr_npi', dest="sw_extr_npi",
                        action='append', type=int,
                        help="Number of extracted subwindows per image "
                             "(if 'random' mode)")

    parser.add_argument('--sw_colorspace', dest="sw_colorspace",
                        type=str, default='RGB__rgb',
                        help="List of colorspace features")
    return parser


def check_params_subwindows(params):
    params.sw_input_size = check_default(params.sw_input_size, 4)
    params.sw_input_size = [(s, s) for s in params.sw_input_size]
    params.sw_output_size = check_default(params.sw_output_size, 1)
    params.sw_output_size = [(s, s) for s in params.sw_output_size]

    params.sw_extr_mode = check_default(params.sw_extr_mode, 'scoremap_constrained', return_list=False)
    params.sw_extr_ratio = check_default(params.sw_extr_ratio, 0.5)
    params.sw_extr_score_thres = check_default(params.sw_extr_score_thres, 0.4)
    params.sw_extr_stride = check_default(params.sw_extr_stride, 1)
    params.sw_extr_npi = check_default(params.sw_extr_npi, 100)

    params.sw_colorspace = params.sw_colorspace.split(' ')

    return params


def parser_randomizedtrees(parser):
    parser.add_argument('--forest_method', dest='forest_method', type=str,
                        action='append', choices=['ET-clf', 'ET-regr', 'RF-clf', 'RF-regr', 'baseline'],
                        help="Type of forest method")
    parser.add_argument('--forest_n_estimators', dest='forest_n_estimators',
                        action='append', type=int,
                        help="Number of trees in forest")
    parser.add_argument('--forest_min_samples_split', dest='forest_min_samples_split',
                        action='append', type=int,
                        help="Minimum number of samples for further splitting")
    parser.add_argument('--forest_max_features', dest='forest_max_features',
                        action='append',
                        help="Max features")
    return parser


def check_params_randomizedtrees(params):
    params.forest_method = check_default(params.forest_method, 'ET-regr')
    params.forest_n_estimators = check_default(params.forest_n_estimators, 1)
    params.forest_min_samples_split = check_default(params.forest_min_samples_split, 2)
    params.forest_max_features = check_default(params.forest_max_features, 'sqrt')
    params.forest_max_features = check_max_features(params.forest_max_features)
    return params


def parser_cnn(parser):
    parser.add_argument('--cnn_architecture', '--architecture', dest='architecture',
                        type=str,
                        choices=['FCRN-A', 'FCRN-B', 'FCRN-test'],
                        help="")
    parser.add_argument('--cnn_initializer', '--initializer', dest='initializer',
                        action='append', type=str,
                        help="")
    parser.add_argument('--cnn_regularizer', '--regularizer', dest='regularizer',
                        action='append', type=str,
                        help="")
    parser.add_argument('--cnn_batch_normalization', '--batch_normalization', dest='batch_normalization',
                        action='append', type=bool,
                        help="")
    parser.add_argument('--cnn_learning_rate', '--learning_rate', '--lr', dest='learning_rate',
                        action='append', type=float,
                        help="")
    parser.add_argument('--cnn_momentum', '--momentum', dest='momentum',
                        action='append', type=float,
                        help="")
    parser.add_argument('--cnn_nesterov', '--nesterov', dest='nesterov',
                        action='append', type=bool,
                        help="")
    parser.add_argument('--cnn_decay', '--decay', dest='decay',
                        action='append', type=float,
                        help="")
    parser.add_argument('--cnn_epochs', '--epochs', dest='epochs',
                        action='append', type=int,
                        help="")
    parser.add_argument('--cnn_batch_size', '--batch_size', dest='batch_size',
                        action='append', type=int,
                        help="")
    return parser


def check_params_cnn(params):
    params.architecture = check_default(params.architecture, 'FCRN-test', return_list=False)
    params.initializer = check_default(params.initializer, 'orthogonal')
    params.regularizer = check_default(params.regularizer, None)
    params.batch_normalization = check_default(params.batch_normalization, True)
    params.learning_rate = check_default(params.learning_rate, 0.02)
    params.momentum = check_default(params.momentum, 0.9)
    params.nesterov = check_default(params.nesterov, True)
    params.decay = check_default(params.decay, 0.)
    params.epochs = check_default(params.epochs, 3)
    params.batch_size = check_default(params.batch_size, 2)

    if 'FCRN' in params.architecture:
        if params.architecture == 'FCRN-A':
            div = 8
        elif params.architecture == 'FCRN-B':
            div = 4
        else:
            div = 1

        new_sw_size = []
        for sis in params.sw_input_size:
            w, h = sis
            new_sw_size.append(((w//div * div + div), (h//div * div + div)))

        params.sw_input_size = new_sw_size
        params.sw_output_size = params.sw_input_size
        params.sw_size = params.sw_input_size

    return params


def parser_augmentation(parser):
    parser.add_argument('--augmentation', dest='augmentation', type=bool,
                        help="")
    parser.add_argument('--aug_rotation_range', dest='rotation_range', type=float,
                        help="")
    parser.add_argument('--aug_width_shift_range', dest='width_shift_range', type=float,
                        help="")
    parser.add_argument('--aug_height_shift_range', dest='height_shift_range', type=float,
                        help="")
    parser.add_argument('--aug_zoom_range', dest='zoom_range', type=float,
                        help="")
    parser.add_argument('--aug_fill_mode', dest='fill_mode', type=str,
                        help="")
    parser.add_argument('--aug_horizontal_flip', dest='horizontal_flip', type=bool,
                        help="")
    parser.add_argument('--aug_vertical_flip', dest='vertical_flip', type=bool,
                        help="")
    parser.add_argument('--aug_featurewise_center', dest='featurewise_center', type=bool,
                        help="")
    parser.add_argument('--aug_featurewise_std_normalization', dest='featurewise_std_normalization', type=bool,
                        help="")
    return parser


def check_params_augmentation(params, default_augmentation=False):
    params.augmentation = check_default(params.augmentation, default_augmentation, return_list=False)
    if params.augmentation:
        params.rotation_range = check_default(params.rotation_range, 30., return_list=False)
        params.width_shift_range = check_default(params.width_shift_range, 0.3, return_list=False)
        params.height_shift_range = check_default(params.height_shift_range, 0.3, return_list=False)
        params.zoom_range = check_default(params.zoom_range, 0.3, return_list=False)
        params.fill_mode = check_default(params.fill_mode, 'constant', return_list=False)
        params.horizontal_flip = check_default(params.horizontal_flip, True, return_list=False)
        params.vertical_flip = check_default(params.vertical_flip, True, return_list=False)
        params.featurewise_center = check_default(params.featurewise_center, False, return_list=False)
        params.featurewise_std_normalization = check_default(params.featurewise_std_normalization, False,
                                                             return_list=False)
    else:
        params.rotation_range = 0.
        params.width_shift_range = 0.
        params.height_shift_range = 0.
        params.zoom_range = 0.
        params.fill_mode = 'reflect'
        params.horizontal_flip = False
        params.vertical_flip = False
        params.featurewise_center = False
        params.featurewise_std_normalization = False
    return params


def parser_cv(parser):
    parser.add_argument('--cv_mode', dest='cv_mode', type=str,
                        choices=['GKF', 'LPGO'],
                        help="Group K-Fold (GKF) or Leave P Groups out (LPGO)")
    parser.add_argument('--cv_param', dest='cv_param',
                        type=int, default=3,
                        help="K if GKF or P if LPGO")
    parser.add_argument('--cv_test_images', dest='cv_test_images',
                        action='append', type=int,
                        help="Images of which the annotations should be the test set")
    parser.add_argument('--cv_epsilon', dest='cv_epsilon',
                        type=int, default=8,
                        help="Maximum distance between a GT and a prediction to be a TP")
    parser.add_argument('--cv_scoring_rank', dest='cv_scoring_rank',
                        type=str, default='f1_score',
                        choices=['f1_score', 'precision_score', 'recall_score',
                                 'accuracy_score'],
                        help="Scoring criterion: f1, accuracy, precision, recall")
    parser.add_argument('--cv_save_to', dest='cv_save_to', type=str,
                        default=None, help="")

    return parser


def check_params_cv(params):
    if params.cv_save_to is None:
        params.cv_save_to = os.path.join(params.cytomine_working_path, "results")
    make_dirs(params.cv_save_to, remove_filename=True)
    return params
