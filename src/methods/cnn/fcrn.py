# -*- coding: utf-8 -*-
import copy
import inspect
import itertools
from collections import defaultdict
from functools import partial

import cv2
import numpy as np
import pandas as pd
import time
import types
from cytomine import Cytomine
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import rankdata
from sklearn.model_selection import ParameterGrid
from sklearn.utils.fixes import MaskedArray
from sldc import DefaultTileBuilder
from tabulate import tabulate
from joblib import Parallel, delayed, logger as joblib_logger

from data.io import open_image, open_scoremap
from data.region_of_interest import load_dataset
from features.postprocessing import non_maximum_suppression
from features.subwindows import mk_subwindows
from jobs.jobs import CytomineJob, LocalJob
from jobs.logger import StandardOutputLogger, Logger
from methods.cnn.architectures import FCRN_A, FCRN_B, sgd_compile, FCRN_test
from softwares.parser import remove_list
from validation.cross_validation import GridSearchCV
from validation.metrics import MetricsEvaluator
from validation.utils import mk_tt_split, cv_strategy, mk_param_grid, MyImage, merge_dicts

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"


def lr_scheduler(epoch):
    step = 24
    num = epoch % step
    if num == 0 and epoch != 0:
        lr_scheduler.lrate = lr_scheduler.lrate - lr_scheduler.lrate / 2.

        # lrate = initial_lrate * 1/(1 + decay * (epoch - num * step))
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr_scheduler.lrate))
    return np.float(lr_scheduler.lrate)


class FCRN(KerasRegressor):
    def __init__(self, build_fn=None, callbacks=None, **sk_params):
        super(FCRN, self).__init__(build_fn, **sk_params)
        self.callbacks = callbacks

    def check_params(self, params):
        """Checks for user typos in "params".

        # Arguments
            params: dictionary; the parameters to be checked

        # Raises
            ValueError: if any member of `params` is not a valid argument.
        """
        legal_params_fns = [Sequential.fit, Sequential.predict,
                            Sequential.predict_classes, Sequential.evaluate,
                            ImageDataGenerator.__init__,
                            mk_subwindows]
        if self.build_fn is None:
            legal_params_fns.append(self.__call__)
        elif (not isinstance(self.build_fn, types.FunctionType) and
                  not isinstance(self.build_fn, types.MethodType)):
            legal_params_fns.append(self.build_fn.__call__)
        else:
            legal_params_fns.append(self.build_fn)

        legal_params = []
        for fn in legal_params_fns:
            legal_params += inspect.getargspec(fn)[0]
        legal_params = set(legal_params)

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

    def get_params(self, **params):
        res = super(FCRN, self).get_params(**params)
        res.update({'callbacks': self.callbacks})
        return res

    def fit(self, x, y, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where n_samples in the number of samples
                and n_features is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for X.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
        # Returns
            history : object
                details about the training history at each epoch.
        """
        self.sk_params['callbacks'] = self.callbacks
        self.sk_params['verbose'] = 2

        lr_scheduler.lrate = self.sk_params['learning_rate']

        if self.build_fn is None:
            self.__model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
                  not isinstance(self.build_fn, types.MethodType)):
            self.__model = self.build_fn(
                    **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.__model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.__model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)
        del fit_args['batch_size']

        # Make subwindows for training
        _x, _y = mk_subwindows(x, y, None, flatten=False, **self.filter_sk_params(mk_subwindows))
        # cv2.imwrite('im.png', np.asarray(_x[0], dtype=np.uint))
        # cv2.imwrite('mask.png', np.asarray(_y[0] * 255, dtype=np.uint))
        _y = np.expand_dims(_y, axis=4)

        print _y
        print _y.shape

        # Generator
        seed = np.random.randint(2 ** 32 - 1)
        exceptions_y_datagen = ['featurewise_center', 'samplewise_center',
                                'featurewise_std_normalization', 'samplewise_std_normalization']
        X_datagen = ImageDataGenerator(**self.filter_sk_params(ImageDataGenerator.__init__))
        y_datagen = ImageDataGenerator(**self.filter_sk_params(ImageDataGenerator.__init__,
                                                               exceptions=exceptions_y_datagen))

        X_datagen.fit(_x, augment=True, seed=seed)
        y_datagen.fit(_y, augment=True, seed=seed)
        X_gen = X_datagen.flow(_x, None, batch_size=self.sk_params['batch_size'], seed=seed)
        y_gen = y_datagen.flow(_y, None, batch_size=self.sk_params['batch_size'], seed=seed)
        datagen = itertools.izip(X_gen, y_gen)

        self.__history = self.__model.fit_generator(
                datagen,
                steps_per_epoch=_x.shape[0] / self.sk_params['batch_size'],
                **fit_args)

        return self.__history

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where n_samples in the number of samples
                and n_features is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.predict`.

        # Returns
            preds: array-like, shape `(n_samples,)`
                Predictions.
        """
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)

        ret_lst = []
        for x in X:
            _X = []
            x = open_image(x, flag='RGB')
            ret = np.zeros((x.shape[0], x.shape[1]))
            count = np.zeros((x.shape[0], x.shape[1]))
            mi = MyImage(x)
            dtb = DefaultTileBuilder()
            tile_iterator = mi.tile_iterator(dtb, max_width=512, max_height=512, overlap=30)
            for tile in tile_iterator:
                height = tile.width
                top = tile.offset_x
                bottom = top + height

                width = tile.height
                left = tile.offset_y
                right = left + width

                tile2 = x[top:bottom, left:right]

                div = 8
                x2 = cv2.copyMakeBorder(tile2, 0, ((height//div * div + div) - height),
                                       0, ((width//div * div + div) - width), borderType=cv2.BORDER_DEFAULT)
                x2 = np.expand_dims(x2, axis=0)
                # print x2.shape
                # print height
                # print width

                pred_tile = self.model.predict(x2, **kwargs)

                # print pred_tile.shape

                add = np.squeeze(pred_tile)[:height, :width]
                ret[top:bottom, left:right] = ret[top:bottom, left:right] + add
                count[top:bottom, left:right] += 1
                ret[count > 1] = ret[count > 1] / count[count > 1]
            ret_lst.append(ret)

        # print np.squeeze(ret_lst).shape
        return np.squeeze(ret_lst)
    
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
        _store('accuracy_score', all_ret[:, 0],  rank=True, weights=test_sample_counts)
        _store('precision_score', all_ret[:, 1],  rank=True, weights=test_sample_counts)
        _store('recall_score', all_ret[:, 2],  rank=True, weights=test_sample_counts)
        _store('f1_score', all_ret[:, 3],  rank=True, weights=test_sample_counts)
        _store('distance_mae', all_ret[:, 4],  rank=True, weights=test_sample_counts)
        _store('count_mae', all_ret[:, 5],  rank=True, weights=test_sample_counts)
        _store('count_pct_mae', all_ret[:, 6],  rank=True, weights=test_sample_counts)
        _store('raw_count_mae', all_ret[:, 7],  rank=True, weights=test_sample_counts)
        _store('raw_count_pct_mae', all_ret[:, 8],  rank=True, weights=test_sample_counts)
        _store('density_mae', all_ret[:, 9],  rank=True, weights=test_sample_counts)
        _store('raw_density_mae', all_ret[:, 10],  rank=True, weights=test_sample_counts)

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
            

    def postprocessing(self, X, **post_params):
        return np.squeeze([non_maximum_suppression(x, **post_params) for x in X])

    @property
    def history(self):
        return self.__history

    @property
    def model(self):
        return self.__model

    @staticmethod
    def build_fcrn(architecture='FCRN-test', regularizer=None, initializer='orthogonal',
                   batch_normalization=False, learning_rate=0.01, momentum=0.9, decay=0.,
                   nesterov=False, input_shape=(None, None, 3)):
        if architecture == 'FCRN-A':
            arch = FCRN_A(input_shape, regularizer, initializer,
                          batch_normalization)
        elif architecture == 'FCRN-B':
            arch = FCRN_B(input_shape, regularizer, initializer,
                          batch_normalization)
        elif architecture == 'FCRN-test':
            arch = FCRN_test(input_shape, regularizer, initializer,
                             batch_normalization)
        else:
            raise ValueError('Unknown method.')

        model = sgd_compile(arch, learning_rate, momentum, decay, nesterov)
        model.summary()
        return model


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

        # Callbacks
        checkpoint_callback = ModelCheckpoint(parameters.model_file, monitor='loss', save_best_only=True)
        lr_callback = LearningRateScheduler(lr_scheduler)
        callbacks = [checkpoint_callback, lr_callback]

        if mk_cv:
            refit = (len(parameters.cv_test_images) > 0)
            # # TODO
            # nontrainable_param_grid = {
            #     'post_threshold': [0.5,0.6], #np.arange(0.0, 1.0, 0.02),
            #     'post_sigma': [0.0, 1., 4.]
            # }
            nontrainable_param_grid = mk_param_grid(vars(parameters), ['post_sigma'])
            nontrainable_param_grid['post_threshold'] = np.arange(0.0, 1.0, 0.02)

            tunable_parameters = ['pre_alpha',
                       'sw_size', 'sw_extr_stride', 'sw_extr_score_thres', 'sw_extr_ratio',
                       'sw_extr_npi', 'initializer', 'regularizer', 'batch_normalization',
                                  'learning_rate', 'momentum', 'decay',
                       'epochs', 'batch_size', 'nesterov']
            param_grid = mk_param_grid(vars(parameters), tunable_parameters)
            print param_grid
            me = MetricsEvaluator(parameters.cv_epsilon, raw_factor=100)

            estimator = GridSearchCV(default_estimator=FCRN(FCRN.build_fcrn, callbacks, **vars(remove_list(parameters))),
                                     param_grid=param_grid, cv=cv_strategy(parameters),
                                     me=me,
                                     nontrainable_param_grid=nontrainable_param_grid, scoring_rank=parameters.cv_scoring_rank,
                                     logger=logger, iid=parameters.image_as_roi,
                                     refit=refit, n_jobs=1) # TODO
            estimator.fit(X, y, labels)
            df = pd.DataFrame(estimator.cv_results_)
            print tabulate(df, headers='keys', tablefmt='grid')
            df.to_csv(parameters.cv_save_to)
            print parameters.cv_save_to

            if refit:
                job.set_progress(status_comment="Performing (best) model validation", progress=90)
                df = pd.DataFrame(estimator.best_estimator_.score(X_test, y_test, me, nontrainable_param_grid))
                print tabulate(df, headers='keys', tablefmt='grid')
                f = parameters.cv_save_to[:-4]+'-assessment.csv'
                df.to_csv(f)
                print f
                print estimator.best_params_
        else:
            estimator = FCRN(FCRN.build_fcrn, callbacks, **vars(parameters))
            estimator.fit(X, y)

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
        model = load_model(parameters.model_file)

        job.set_progress(status_comment="Predicting", progress=20)
        A = model.predict(X_test)

        job.set_progress(status_comment="Computing statistics", progress=80)
        mean_diff = np.average(np.abs(np.sum(np.sum(A, 1), 1) - np.sum(np.sum(y_test, 1), 1))) / 100.0
        print('After training, the difference is : {} cells per image.'.format(np.abs(mean_diff)))
        # TODO metrics

        job.set_progress(status_comment="Finished.", progress=100)
