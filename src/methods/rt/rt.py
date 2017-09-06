# -*- coding: utf-8 -*-
from collections import defaultdict
from functools import partial

import itertools
import numpy as np
import pandas as pd
import time
import types
from cytomine import Cytomine
from scipy.stats import rankdata
from sklearn.dummy import DummyRegressor
from sklearn.ensemble.forest import (
    ForestClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.model_selection import ParameterGrid
from sklearn.utils.fixes import MaskedArray
from tabulate import tabulate

from data.io import open_image_with_mask, open_image, open_scoremap
from data.region_of_interest import load_dataset
from features.subwindows import mk_subwindows, half_size, all_subwindows_generator, subwindow_box
from jobs.jobs import CytomineJob, LocalJob
from jobs.logger import Logger, StandardOutputLogger
from methods.base_method import BaseMethod
from softwares.parser import remove_list
from validation.cross_validation import GridSearchCV
from validation.metrics import MetricsEvaluator
from validation.utils import mk_tt_split, mk_param_grid, cv_strategy
from joblib import Parallel, delayed, logger as joblib_logger

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"


class CellCountRandomizedTrees(BaseMethod):
    def __init__(self, build_fn=None, logger=StandardOutputLogger(Logger.INFO), **sk_params):
        super(CellCountRandomizedTrees, self).__init__(build_fn, logger, **sk_params)
        self.build_fn = build_fn
        self.sk_params = sk_params
        self.logger = logger

    def fit(self, X, y, _X=None, _y=None):
        if self.build_fn is None:
            self.__forest = self.build_rt(**self.filter_sk_params(self.build_rt))
        elif (not isinstance(self.build_fn, types.FunctionType) and
                  not isinstance(self.build_fn, types.MethodType)):
            self.__forest = self.build_fn(
                    **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.__forest = self.build_fn(**self.filter_sk_params(self.build_fn))

        X, y = np.asarray(X), np.asarray(y)
        if _X is None and _y is None:
            _X, _y = self.extract_subwindows(X, y)

        self.logger.i('[FIT] Start fitting from {} images, {} subwindows'.format(X.shape[0], _X.shape[0]))
        self.__forest.fit(_X, _y)

        if isinstance(self.__forest, ForestClassifier):
            self.foreground_class_ = np.where(self.__forest.classes_ == 1.)

        return self

    def extract_subwindows(self, X, y, labels=None):
        X, y = np.asarray(X), np.asarray(y)

        self.logger.i("[EXTRACT SUBWINDOWS] Start extracting subwindows from {} images".format(X.shape[0]))
        _X, _y = mk_subwindows(X, y, labels, **self.filter_sk_params(mk_subwindows))

        self.logger.i("[EXTRACT SUBWINDOWS] _X size: ({} samples / {} features)".format(_X.shape[0], _X.shape[1]))
        self.logger.i("[EXTRACT SUBWINDOWS] _y size: ({} samples)".format(_y.shape[0]))
        return _X, _y

    def _predict_clf_helper(self, _x, foreground_class):
        foreground_class = np.asarray(foreground_class).squeeze()
        return self.__forest.predict_proba(_x)[:, foreground_class]

    def predict(self, X):
        """Returns predictions for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where n_samples in the number of samples
                and n_features is the number of features.

        # Returns
            preds: array-like, shape `(n_samples,)`
                Predictions.
        """
        self.logger.i("[PREDICT]")
        ret_lst = []
        for x in X:
            if hasattr(self, 'foreground_class_'):
                predict_method = partial(self._predict_clf_helper,
                                         foreground_class=self.foreground_class_)
            else:
                predict_method = self.__forest.predict

            window_input_size_half = half_size(self.sk_params['sw_input_size'])
            window_output_size_half = half_size(self.sk_params['sw_output_size'])
            image, mask = open_image_with_mask(x, padding=window_input_size_half)
            y = np.zeros_like(mask, dtype=np.float16)
            count = np.zeros_like(mask, dtype=np.uint16)

            asg = all_subwindows_generator(image, mask, batch_size= image.shape[0] * image.shape[1],
                                           **self.filter_sk_params(all_subwindows_generator))
            for sws, coords in asg:
                predictions = predict_method(sws)
                for prediction, coord in zip(predictions, coords):
                    top, right, bottom, left = subwindow_box(self.sk_params['sw_output_size'],
                                                             window_output_size_half, coord)
                    y[slice(top, bottom), slice(left, right)] += prediction.reshape(
                            self.sk_params['sw_output_size'])
                    count[slice(top, bottom), slice(left, right)] += 1

            y[count > 1] = y[count > 1] / count[count > 1]

            ret_lst.append(y[window_input_size_half[0]: -window_input_size_half[0],
                           window_input_size_half[1]: -window_input_size_half[1]])

        return np.squeeze(ret_lst)

    @property
    def forest(self):
        return self.__forest

    @staticmethod
    def build_rt(forest_method, forest_n_estimators, forest_min_samples_split, forest_max_features, n_jobs):
        if 'baseline' in forest_method:
            LearningMethod = partial(DummyRegressor, strategy='median')
            return LearningMethod()
        elif 'ET' in forest_method:
            if 'clf' in forest_method:
                LearningMethod = partial(ExtraTreesClassifier, class_weight='balanced')
            else:
                LearningMethod = ExtraTreesRegressor
        else:
            if 'clf' in forest_method:
                LearningMethod = partial(RandomForestClassifier, class_weight='balanced')
            else:
                LearningMethod = RandomForestRegressor

        return LearningMethod(
                n_estimators=forest_n_estimators,
                min_samples_split=forest_min_samples_split,
                max_features=forest_max_features,
                n_jobs=n_jobs)

    def score(self, X_test, y_test, me, nontrainable_param_grid):
        start_time = time.time()
        candidate_nontrainable_params = ParameterGrid(nontrainable_param_grid)
        n_nontrainable_candidates = len(candidate_nontrainable_params)
        all_ret = []
        for nontrainable_parameters in candidate_nontrainable_params:
            me.reset()
            for x, y in itertools.izip(X_test, y_test):
                p = self.predict(np.array([x]))
                pp = self.postprocessing([p], **nontrainable_parameters)
                me.compute([open_scoremap(y)], [pp], [p])
            metrics = me.all_metrics()
            score_time = time.time() - start_time

            # msg += ", score=%f" % test_score
            total_time = score_time
            msg1 = '%s' % (', '.join('%s=%s' % (k, v)
                                     for k, v in nontrainable_parameters.items()))
            end_msg = "%s total=%s" % (msg1, joblib_logger.short_format_time(total_time))
            print ("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

            ret = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['distance'],
                   metrics['count'], metrics['count_pct'], metrics['raw_count'], metrics['raw_count_pct'],
                   metrics['density'], metrics['raw_density'], X_test.shape[0]]

            all_ret.append(ret)
        all_ret = np.asarray(all_ret)

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            array = np.array(array, dtype=np.float64).reshape(1,
                                                              n_nontrainable_candidates).T

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                        rankdata(-array_means, method='min'), dtype=np.int32)

        test_sample_counts = all_ret[:, 11]
        test_sample_counts = np.array(test_sample_counts[::n_nontrainable_candidates], dtype=np.int)
        _store('accuracy_score', all_ret[:, 0], rank=True, weights=test_sample_counts)
        _store('precision_score', all_ret[:, 1], rank=True, weights=test_sample_counts)
        _store('recall_score', all_ret[:, 2], rank=True, weights=test_sample_counts)
        _store('f1_score', all_ret[:, 3], rank=True, weights=test_sample_counts)
        _store('distance_mae', all_ret[:, 4], rank=True, weights=test_sample_counts)
        _store('count_mae', all_ret[:, 5], rank=True, weights=test_sample_counts)
        _store('count_pct_mae', all_ret[:, 6], rank=True, weights=test_sample_counts)
        _store('raw_count_mae', all_ret[:, 7], rank=True, weights=test_sample_counts)
        _store('raw_count_pct_mae', all_ret[:, 8], rank=True, weights=test_sample_counts)
        _store('density_mae', all_ret[:, 9], rank=True, weights=test_sample_counts)
        _store('raw_density_mae', all_ret[:, 10], rank=True, weights=test_sample_counts)

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_nontrainable_candidates, ),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(list(candidate_nontrainable_params)):
            # params = merge_dicts(*params)
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        return results


def train(parameters, mk_cv=False):
    # Initialize logger
    logger = StandardOutputLogger(parameters.verbose)
    for key, val in sorted(vars(parameters).iteritems()):
        logger.info("[PARAMETER] {}: {}".format(key, val))

    # Initialize Cytomine client
    cytomine = Cytomine(
            parameters.cytomine_host,
            parameters.cytomine_public_key,
            parameters.cytomine_private_key,
            working_path=parameters.cytomine_working_path,
            base_path=parameters.cytomine_base_path,
            verbose=(parameters.verbose >= Logger.DEBUG)
    )

    # Start job
    with CytomineJob(cytomine,
                     parameters.cytomine_software,
                     parameters.cytomine_project,
                     parameters=vars(parameters)) \
            if parameters.cytomine_software is not None \
            else LocalJob() as job:

        job.logger = logger
        job.set_progress(status_comment="Starting...", progress=0)

        job.set_progress(status_comment="Loading dataset...", progress=1)
        X, y, labels = load_dataset(parameters, cytomine)
        logger.d("X size: {} samples".format(X.shape[0]))
        logger.d("y size: {} samples".format(y.shape[0]))
        logger.d("labels size: {} samples".format(labels.shape[0]))

        job.set_progress(status_comment="Divide LS and TS used for model validation", progress=2)
        (X, y, labels, X_test, y_test, labels_test) = mk_tt_split(X, y, labels, parameters.cv_test_images)

        if mk_cv:
            refit = False  # (len(parameters.cv_test_images) > 0)
            # TODO
            nontrainable_param_grid = {
                'post_threshold': np.arange(0.0, 1.0, 0.02),
                'post_sigma': [0.0, 1., 4.]
            }

            tunable_parameters = ['pre_alpha', 'sw_input_size', 'sw_output_size',
                                  'sw_extr_stride', 'sw_extr_score_thres', 'sw_extr_ratio',
                                  'sw_extr_npi', 'forest_method', 'forest_n_estimators',
                                  'forest_min_samples_split', 'forest_max_features']
            param_grid = mk_param_grid(vars(parameters), tunable_parameters)
            print param_grid

            estimator = GridSearchCV(default_estimator=CellCountRandomizedTrees(logger=logger,
                                                                                **vars(remove_list(parameters))),
                                     param_grid=param_grid, cv=cv_strategy(parameters),
                                     me=MetricsEvaluator(parameters.cv_epsilon), # TODO multiply by 100 for density
                                     nontrainable_param_grid=nontrainable_param_grid,
                                     scoring_rank=parameters.cv_scoring_rank,
                                     logger=logger, refit=refit, n_jobs=parameters.n_jobs)
            estimator.fit(X, y, labels)
            df = pd.DataFrame(estimator.cv_results_)
            print tabulate(df, headers='keys', tablefmt='grid')
            df.to_csv(parameters.cv_save_to)
            print parameters.cv_save_to

            if refit:
                job.set_progress(status_comment="Performing (best) model validation", progress=90)
                # TODO
        else:
            estimator = CellCountRandomizedTrees(logger=logger, **vars(parameters))
            estimator.fit(X, y)
            nontrainable_param_grid = {
                'post_threshold': [0.70],
                'post_sigma': [4.]
            }
            job.set_progress(status_comment="Performing (best) model validation", progress=90)
            me=MetricsEvaluator(parameters.cv_epsilon)
            df = pd.DataFrame(estimator.score(X_test, y_test, me, nontrainable_param_grid))
            print tabulate(df, headers='keys', tablefmt='grid')
            f = parameters.cv_save_to[:-4] + '-assessment.csv'
            df.to_csv(f)
            print f

        job.set_progress(status_comment="Saving (best) model", progress=95)
        # estimator.model.save(parameters.model_file)

        job.set_progress(status_comment="Finished.", progress=100)


def predict(parameters):
    # Initialize logger
    logger = StandardOutputLogger(parameters.verbose)
    for key, val in sorted(vars(parameters).iteritems()):
        logger.info("[PARAMETER] {}: {}".format(key, val))

    # Initialize Cytomine client
    cytomine = Cytomine(
            parameters.cytomine_host,
            parameters.cytomine_public_key,
            parameters.cytomine_private_key,
            working_path=parameters.cytomine_working_path,
            base_path=parameters.cytomine_base_path,
            verbose=(parameters.verbose >= Logger.DEBUG)
    )

    # Start job
    with CytomineJob(cytomine,
                     parameters.cytomine_software,
                     parameters.cytomine_project,
                     parameters=vars(parameters)) \
            if parameters.cytomine_software is not None \
            else LocalJob() as job:
        job.logger = logger
        job.set_progress(status_comment="Starting...", progress=0)

        job.set_progress(status_comment="Loading dataset...", progress=1)
        X, y, labels = load_dataset(parameters, cytomine)
        logger.d("X size: {} samples".format(X.shape[0]))
        logger.d("y size: {} samples".format(y.shape[0]))
        logger.d("labels size: {} samples".format(labels.shape[0]))

        job.set_progress(status_comment="Divide LS and TS used for model validation", progress=2)
        (X, y, labels, X_test, y_test, labels_test) = mk_tt_split(X, y, labels, parameters.cv_test_images)
        # TODO Predict takes file name
        X_test = np.array([open_image(x, flag='RGB') for x in X_test])
        y_test = np.expand_dims(np.array([open_scoremap(y) for y in y_test]), axis=4)

        job.set_progress(status_comment="Loading model", progress=10)
        # TODO load model

        job.set_progress(status_comment="Predicting", progress=20)
        # A = model.predict(X_test)

        job.set_progress(status_comment="Computing statistics", progress=80)
        # mean_diff = np.average(np.abs(np.sum(np.sum(A, 1), 1) - np.sum(np.sum(y_test, 1), 1))) / 100.0
        # print('After training, the difference is : {} cells per image.'.format(np.abs(mean_diff)))
        # TODO metrics

        job.set_progress(status_comment="Finished.", progress=100)
