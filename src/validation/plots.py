# -*- coding: utf-8 -*-
import ast
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isfile
from tabulate import tabulate

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"

FIGURES = '../../figures/'


def pr_curve(df, t=0.):
    df = df.sort_values(by='mean_recall_score')
    p = np.asarray(df['mean_precision_score'])
    r = np.asarray(df['mean_recall_score'])
    r_interpolate = np.arange(0., 1.01, 0.01)
    p_interpolate = np.interp(r_interpolate, r, p)
    auc = np.trapz(p_interpolate, r_interpolate)
    return p, r, p_interpolate, r_interpolate, auc


def custom_score(df, alpha=0.5):
    f1 = df['mean_f1_score']
    count = np.asarray(1. - df['mean_count_pct_mae'] / 100.)
    f1 /= np.max(f1)
    count /= np.max(count)
    tot = np.average(np.vstack((f1, count)), axis=0, weights=[alpha, 1-alpha])
    return tot


def plot_pr_curve_by_param(df, infos):
    param_values = np.unique(df['param_{}'.format(infos['param'])])
    post_sigmas = np.unique(df['param_post_sigma'])
    for param_value in param_values:
        plt.figure()
        pdf = df[df['param_{}'.format(infos['param'])] == param_value]
        for post_sigma in post_sigmas:
            sdf = pdf[pdf['param_post_sigma']==post_sigma]
            p, r, p_interpolate, r_interpolate, auc = pr_curve(sdf)
            plt.plot(r, p, label='sigma={}'.format(post_sigma))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim(0.,1.)
            plt.ylim(0.,1.)
            plt.title('{} - {} - PR Curve for {} = {}'.format(infos['dataset'], infos['method'],
                                                              infos['param'], param_value))
            plt.legend()
            print 'AUC for {} = {} and sigma = {} : {}'.format(infos['param'], param_value, post_sigma, auc)

        plt.savefig(os.path.join(FIGURES, infos['dataset']+'_'+infos['method']+'_pr_curve_'+infos['param']+'.png'))

def plot_pr_curve_by_sigma_assessment(df, infos):
    post_sigmas = np.unique(df['param_post_sigma'])

    for post_sigma in post_sigmas:
        plt.figure()
        sdf = df[df['param_post_sigma'] == post_sigma]
        pdf = sdf
        p, r, p_interpolate, r_interpolate, auc = pr_curve(pdf)
        plt.plot(r, p, '.-',label='(AUC={:.4f})'.format(auc))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.title('{} - {} - PR Curve for sigma = {}'.format(infos['dataset'], infos['method'],
                                                          post_sigma))
        plt.legend()
        print 'AUC sigma = {} : {}'.format(post_sigma, auc)

        f = '{}_{}_pr_curve_{}_{}.png'.format(infos['dataset'], infos['method'], post_sigma, infos['param'])
        plt.savefig( os.path.join(FIGURES, f))

def plot_pr_curve_by_sigma(df, infos):
    param_values = np.unique(df['param_{}'.format(infos['param'])])
    post_sigmas = np.unique(df['param_post_sigma'])

    for post_sigma in post_sigmas:
        plt.figure()
        sdf = df[df['param_post_sigma'] == post_sigma]
        for param_value in param_values:
            pdf = sdf[sdf['param_{}'.format(infos['param'])] == param_value]
            p, r, p_interpolate, r_interpolate, auc = pr_curve(pdf)
            plt.plot(r, p, '.-',label='{}={} (AUC={:.4f})'.format(infos['param'], param_value, auc))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim(0., 1.)
            plt.ylim(0., 1.)
            plt.title('{} - {} - PR Curve for sigma = {}'.format(infos['dataset'], infos['method'],
                                                              post_sigma))
            plt.legend()
            print 'AUC for {} = {} and sigma = {} : {}'.format(infos['param'], param_value, post_sigma, auc)

        f = '{}_{}_pr_curve_{}_{}.png'.format(infos['dataset'], infos['method'], post_sigma, infos['param'])
        plt.savefig( os.path.join(FIGURES, f))


def plot_scores_by_sigma(df, infos, scores):
    try:
        param_values = np.unique(df['param_{}'.format(infos['param'])])
    except:
        param_values = [0]
    post_sigmas = np.unique(df['param_post_sigma'])
    for post_sigma in post_sigmas:
        plt.figure()
        sdf = df[df['param_post_sigma'] == post_sigma]

        f1_scores = np.zeros_like(param_values, dtype=np.float)
        p_scores = np.zeros_like(param_values, dtype=np.float)
        r_scores = np.zeros_like(param_values, dtype=np.float)
        acc_scores = np.zeros_like(param_values, dtype=np.float)
        auc_scores = np.zeros_like(param_values, dtype=np.float)
        x = np.zeros_like(param_values, dtype=np.float)

        for i, param_value in enumerate(param_values):
            pdf = sdf[sdf['param_{}'.format(infos['param'])] == param_value]
            _, _, _, _, auc_scores[i] = pr_curve(pdf)

            pdf = pdf.ix[pdf['rank_f1_score'].idxmin()]
            f1_scores[i] = pdf['mean_f1_score']
            p_scores[i] = pdf['mean_precision_score']
            r_scores[i] = pdf['mean_recall_score']
            acc_scores[i] = pdf['mean_accuracy_score']
            try:
                param_value = np.float(param_value)
            except:
                param_value = ast.literal_eval(param_value)

            if isinstance(param_value, tuple):
                x[i] = param_value[0]
            else:
                x[i] = param_value

        ordered_idxs = x.argsort()
        x = x[ordered_idxs]

        if 'f1' in scores:
            f1_scores = f1_scores[ordered_idxs]
            plt.plot(x, f1_scores, 'C0.-', label='F1-score')
            f1_max = f1_scores.argmax()
            plt.plot(x[f1_max], f1_scores[f1_max], 'C0*', markersize=10)

        if 'precision' in scores:
            p_scores = p_scores[ordered_idxs]
            plt.plot(x, p_scores, 'C1.-', label="Precision")
            p_max = p_scores.argmax()
            plt.plot(x[p_max], p_scores[p_max], 'C1*', markersize=10)

        if 'recall' in scores:
            r_scores = r_scores[ordered_idxs]
            plt.plot(x, r_scores, 'C2.-', label="Recall")
            r_max = r_scores.argmax()
            plt.plot(x[r_max], r_scores[r_max], 'C2*', markersize=10)

        if 'accuracy' in scores:
            acc_scores = acc_scores[ordered_idxs]
            plt.plot(x, acc_scores, 'C3.-', label="Accuracy")
            acc_max = acc_scores.argmax()
            plt.plot(x[acc_max], acc_scores[acc_max], 'C3*', markersize=10)

        if 'auc' in scores:
            auc_scores = auc_scores[ordered_idxs]
            plt.plot(x, auc_scores, 'C4.-', label="AUC")
            auc_max = auc_scores.argmax()
            plt.plot(x[auc_max], auc_scores[auc_max], 'C4*', markersize=10)

        plt.ylim(0,1)
        plt.xlabel('{}'.format(infos['param']))
        plt.ylabel('Score')
        plt.title('{} - {} - {} - Scores for sigma = {}'.format(infos['dataset'], infos['method'], infos['param'], post_sigma))
        plt.legend()

        f = '{}_{}_score_{}_{}.png'.format(infos['dataset'], infos['method'], post_sigma, infos['param'])
        plt.savefig(os.path.join(FIGURES, f))


def plot_count_by_sigmas(df, infos):
    try:
        param_values = np.unique(df['param_{}'.format(infos['param'])])
    except:
        param_values = [0]
    post_sigmas = np.unique(df['param_post_sigma'])
    for post_sigma in post_sigmas:
        plt.figure()
        sdf = df[df['param_post_sigma'] == post_sigma]

        count_mae = np.zeros_like(param_values, dtype=np.float)
        raw_count_mae = np.zeros_like(param_values, dtype=np.float)
        x = np.zeros_like(param_values, dtype=np.float)

        for i, param_value in enumerate(param_values):
            pdf = sdf[sdf['param_{}'.format(infos['param'])] == param_value]

            pdf = pdf.ix[pdf['rank_f1_score'].idxmin()]
            count_mae[i] = pdf['mean_count_mae']
            raw_count_mae[i] = pdf['mean_raw_count_mae']
            try:
                param_value = np.float(param_value)
            except:
                param_value = ast.literal_eval(param_value)

            if isinstance(param_value, tuple):
                x[i] = param_value[0]
            else:
                x[i] = param_value

        ordered_idxs = x.argsort()
        x = x[ordered_idxs]

        count_mae = count_mae[ordered_idxs]
        plt.plot(x, count_mae, 'C0.-', label='Count MAE')
        count_min = count_mae.argmin()
        plt.plot(x[count_min], count_mae[count_min], 'C0*', markersize=10)

        raw_count_mae = raw_count_mae[ordered_idxs]
        plt.plot(x, raw_count_mae, 'C1.-', label='Raw Count MAE')
        raw_count_min = raw_count_mae.argmin()
        plt.plot(x[raw_count_min], raw_count_mae[raw_count_min], 'C1*', markersize=10)

        plt.xlabel('{}'.format(infos['param']))
        plt.ylabel('Count MAE')
        plt.title('{} - {} - {} - Count MAE for sigma = {}'.format(infos['dataset'], infos['method'], infos['param'],
                                                                   post_sigma))
        plt.legend()

        f = '{}_{}_count_{}_{}.png'.format(infos['dataset'], infos['method'], post_sigma, infos['param'])
        plt.savefig(os.path.join(FIGURES, f))


def plot_threshold_by_sigma(df, infos):
    try:
        param_values = np.unique(df['param_{}'.format(infos['param'])])
    except:
        param_values = [0]
    post_sigmas = np.unique(df['param_post_sigma'])
    for post_sigma in post_sigmas:
        plt.figure()
        sdf = df[df['param_post_sigma'] == post_sigma]

        count_mae = np.zeros_like(param_values, dtype=np.float)
        raw_count_mae = np.zeros_like(param_values, dtype=np.float)
        x = np.zeros_like(param_values, dtype=np.float)

        for i, param_value in enumerate(param_values):
            pdf = sdf[sdf['param_{}'.format(infos['param'])] == param_value]

            pdf = pdf.sort_values(by='param_post_threshold')
            plt.plot(pdf['param_post_threshold'], pdf['mean_f1_score'], '^-', label='F1 {}={}'.format(infos['param'], param_value))
            plt.plot(pdf['param_post_threshold'], pdf['mean_count_pct_mae']/100., '.-', label='Count MAE (%) {}={}'.format(infos['param'], param_value))

        plt.xlabel('Post threshold')
        plt.ylabel('Score')
        plt.ylim(0,1.01)
        plt.grid()
        plt.title('{} - {} - {} - For sigma = {}'.format(infos['dataset'], infos['method'], infos['param'],
                                                                   post_sigma))
        plt.legend()

        f = '{}_{}_threshold_{}_{}.png'.format(infos['dataset'], infos['method'], post_sigma, infos['param'])
        plt.savefig(os.path.join(FIGURES, f))



def plot_threshold_by_sigma2(df, infos):
    param_values = np.unique(df['param_{}'.format(infos['param'])])
    post_sigmas = np.unique(df['param_post_sigma'])
    for post_sigma in post_sigmas:
        plt.figure()
        sdf = df[df['param_post_sigma'] == post_sigma]

        count_mae = np.zeros_like(param_values, dtype=np.float)
        raw_count_mae = np.zeros_like(param_values, dtype=np.float)
        x = np.zeros_like(param_values, dtype=np.float)

        for i, param_value in enumerate(param_values):
            pdf = sdf[sdf['param_{}'.format(infos['param'])] == param_value]

            f1 = pdf['mean_f1_score']
            count = 1 - pdf['mean_count_pct_mae']/100.
            custom = custom_score(pdf)

            pdf = pdf.sort_values(by='param_post_threshold')
            plt.plot(pdf['param_post_threshold'], pdf['mean_f1_score'], '^-', label='F1 {}={}'.format(infos['param'], param_value))
            plt.plot(pdf['param_post_threshold'], pdf['mean_count_pct_mae']/100., '.-', label='Count MAE (%) {}={}'.format(infos['param'], param_value))
            plt.plot(pdf['param_post_threshold'], custom, '.-', label='F1+count normalized {}={}'.format(infos['param'], param_value))

        plt.xlabel('Post threshold')
        plt.ylabel('Score')
        plt.ylim(0,1.01)
        plt.grid()
        plt.title('{} - {} - {} - For sigma = {}'.format(infos['dataset'], infos['method'], infos['param'],
                                                                   post_sigma))
        plt.legend()

        f = '{}_{}_threshold2_{}_{}.png'.format(infos['dataset'], infos['method'], post_sigma, infos['param'])
        plt.savefig(os.path.join(FIGURES, f))


def latex_param(param):
    if param == 'epochs':
        return '$e$'
    elif param == 'batch_size':
        return '$b$'
    elif param == 'learning_rate':
        return '$\gamma$'
    elif param == 'decay':
        return  '$\lambda$'
    elif param == 'sw_extr_npi':
        return  '$N_{sw\ img}$'
    elif param == 'sw_size':
        return '$w\\times h$'

def latex_results(df, infos):
    try:
        param_values = np.unique(df['param_{}'.format(infos['param'])])
    except:
        param_values = [0]
    # post_sigmas = np.array([0.])
    post_sigmas = np.unique(df['param_post_sigma'])

    tab = np.zeros((param_values.shape[0] * post_sigmas.shape[0], 11), dtype=object)
    for j, post_sigma in enumerate(post_sigmas):
        sdf = df[df['param_post_sigma'] == post_sigma]

        for i, param_value in enumerate(param_values):
            pdf = sdf[sdf['param_{}'.format(infos['param'])] == param_value]
            _, _, _, _, auc = pr_curve(pdf)

            custom = custom_score(pdf)
            pdf = pdf.iloc[np.argmax(custom)]
            custom = np.round(np.max(custom), decimals=2)
            f1 = np.round(pdf['mean_f1_score'], decimals=4)
            p = np.round(pdf['mean_precision_score'], decimals=2)
            r = np.round(pdf['mean_recall_score'], decimals=2)
            # acc = np.round(pdf['mean_accuracy_score'], decimals=2)
            count = np.round(pdf['mean_count_mae'], decimals=2)
            count_pct = np.round(pdf['mean_count_pct_mae'], decimals=2)
            raw_count = np.round(pdf['mean_raw_count_mae'], decimals=2)
            raw_count_pct = np.round(pdf['mean_raw_count_pct_mae'], decimals=2)
            dist = np.round(pdf['mean_distance_mae'], decimals=2)



            tab[j * param_values.shape[0] + i,:] = [infos['param'], param_value, post_sigma, p, r, f1,
                                                count, count_pct, raw_count, raw_count_pct, dist]
            ordered_idxs = tab[:,1].argsort()

    print tabulate(tab[ordered_idxs], tablefmt='latex')



def latex_results_assessment(df, infos):
    post_sigmas = np.unique(df['param_post_sigma'])

    tab = np.zeros((post_sigmas.shape[0], 11), dtype=object)
    for j, post_sigma in enumerate(post_sigmas):
        sdf = df[df['param_post_sigma'] == post_sigma]

        pdf = sdf
        _, _, _, _, auc = pr_curve(pdf)

        custom = custom_score(pdf)
        pdf = pdf.iloc[np.argmax(custom)]
        custom = np.round(np.max(custom), decimals=2)
        f1 = np.round(pdf['mean_f1_score'], decimals=4)
        p = np.round(pdf['mean_precision_score'], decimals=2)
        r = np.round(pdf['mean_recall_score'], decimals=2)
        # acc = np.round(pdf['mean_accuracy_score'], decimals=2)
        count = np.round(pdf['mean_count_mae'], decimals=2)
        count_pct = np.round(pdf['mean_count_pct_mae'], decimals=2)
        raw_count = np.round(pdf['mean_raw_count_mae'], decimals=2)
        raw_count_pct = np.round(pdf['mean_raw_count_pct_mae'], decimals=2)
        dist = np.round(pdf['mean_distance_mae'], decimals=2)

        tab[j, :] = [infos['dataset'], '-', post_sigma, p, r, f1,
                                                 count, count_pct, raw_count, raw_count_pct, dist]
        ordered_idxs = tab[:, 2].argsort()

    print tabulate(tab[ordered_idxs], tablefmt='latex')



def plots_from_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    name = os.path.splitext(os.path.basename(csv_file))[0]
    name_split = name.split('_')
    infos = dict()
    infos['dataset'] = name_split[0]
    infos['method'] = name_split[1]
    infos['param'] = '_'.join(name_split[2:])

    print '#' * len(name)
    print name
    print '#' * len(name)

    # plot_pr_curve_by_param(df, infos)
    plot_pr_curve_by_sigma(df, infos)
    # plot_scores_by_sigma(df, infos, ['f1', 'precision', 'recall', 'accuracy', 'auc'])
    # plot_count_by_sigmas(df, infos)
    # plot_threshold_by_sigma2(df, infos)
    plot_threshold_by_sigma(df, infos)
    latex_results(df, infos)


    # plt.show()
    plt.close()


def plot_assessment_from_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    name = os.path.splitext(os.path.basename(csv_file))[0]
    name_split = name.split('_')
    infos = dict()
    infos['dataset'] = name_split[0]
    infos['method'] = name_split[1]
    infos['param'] = '_'.join(name_split[2:])
    plot_pr_curve_by_sigma_assessment(df, infos)

    print '#' * len(name)
    print name
    print '#' * len(name)
    latex_results_assessment(df, infos)

DEFAULT_PARAMS = {
    'e': 100,
}


def best_nontrainable(df, param, default_model):
    param_values = np.unique(df['param_{}'.format(param)])

    results = np.zeros((param_values.shape[0], 10), dtype=object)

    for i, param_value in enumerate(param_values):
        pdf = df[df['param_{}'.format(param)] == param_value]
        _, _, _, _, auc = pr_curve(pdf)

        custom = custom_score(pdf)
        pdf = pdf.iloc[np.argmax(custom)]
        custom = np.round(np.max(custom), decimals=2)
        f1 = np.round(pdf['mean_f1_score'], decimals=4)
        p = np.round(pdf['mean_precision_score'], decimals=2)
        r = np.round(pdf['mean_recall_score'], decimals=2)
        acc = np.round(pdf['mean_accuracy_score'], decimals=2)
        count = np.round(pdf['mean_count_mae'], decimals=2)
        count_pct = np.round(pdf['mean_count_pct_mae'], decimals=2)
        raw_count = np.round(pdf['mean_raw_count_mae'], decimals=2)
        raw_count_pct = np.round(pdf['mean_raw_count_pct_mae'], decimals=2)
        dist = np.round(pdf['mean_distance_mae'], decimals=2)

        results[i,:] = [param, param_value, pdf['param_post_threshold'], pdf['param_post_sigma'],
                    f1, count, count_pct, raw_count, raw_count_pct, dist]

    # Sort by parameter value
    ordered_idxs = results[:,1].argsort()
    return results[ordered_idxs]

def best_gs(gs, sigma=None, kappa=None):


    tab = np.zeros((1, 10), dtype=object)

    pdf = gs
    _, _, _, _, auc = pr_curve(pdf)

    custom = custom_score(pdf)

    if sigma is None or kappa is None:
        pdf = pdf.iloc[np.argmax(custom)]
    else:
        pdf = pdf[pdf['param_post_sigma']==sigma]
        pdf = pdf[pdf['param_post_threshold']==kappa]
    custom = np.round(np.max(custom), decimals=2)
    f1 = np.round(pdf['mean_f1_score'], decimals=4)
    p = np.round(pdf['mean_precision_score'], decimals=2)
    r = np.round(pdf['mean_recall_score'], decimals=2)
    # acc = np.round(pdf['mean_accuracy_score'], decimals=2)
    count = np.round(pdf['mean_count_mae'], decimals=2)
    count_pct = np.round(pdf['mean_count_pct_mae'], decimals=2)
    raw_count = np.round(pdf['mean_raw_count_mae'], decimals=2)
    raw_count_pct = np.round(pdf['mean_raw_count_pct_mae'], decimals=2)
    dist = np.round(pdf['mean_distance_mae'], decimals=2)

    tab[0, :] = [pdf['param_post_threshold'], pdf['param_post_sigma'],
                    p, r, f1, count, count_pct, raw_count, raw_count_pct, dist]
    ordered_idxs = tab[:, 2].argsort()

    try:
        print 'epochs '+ str(pdf['param_epochs'])
        print 'batch ' + str(pdf['param_batch_size'])
        print 'lr ' + str(pdf['param_learning_rate'])
        print 'decay '+ str(pdf['param_decay'])
        print 'npi ' + str(pdf['param_sw_extr_npi'])
        print 'size ' + str(pdf['param_sw_size'])
    except:
        pass

    # Sort by parameter value
    ordered_idxs = tab[:, 1].argsort()
    return tab[ordered_idxs]


def model_selection_latex(dataset, method, files):
    dfs = []
    for f in files:
        if 'assessment' not in f and dataset in f and method in f:
            name = os.path.splitext(os.path.basename(f))[0]
            name_split = name.split('_')
            dfs.append({'param': '_'.join(name_split[2:]),
                        'df': pd.read_csv(f)
                        })

    # Compute default model
    default_model = {}

    if len(dfs) > 0:
        results = np.concatenate([best_nontrainable(df['df'], df['param'], default_model) for df in dfs])
        print tabulate(results, tablefmt='latex')


SIGMAS = {
        'BMGRAZ': {'SRTC': 4., 'PRTR': 1., 'DRTR': 4, 'FCRN-A': 0., 'FCRN-B': 4., 'baseline':0.},
        'ANAPATH': {'SRTC': 0., 'PRTR': 0., 'DRTR': 0., 'FCRN-A': 5., 'FCRN-B': 5.},
        'GANGLIONS': {'SRTC': 1., 'PRTR': 1., 'DRTR': 0., 'FCRN-A': 0., 'FCRN-B': 1.},
        'CRC': {'SRTC': 0., 'PRTR': 0., 'DRTR': 0., 'FCRN-A': 1., 'FCRN-B': 1.}
    }

KAPPAS = {
        'BMGRAZ': {'SRTC': 0.78, 'PRTR': .64, 'DRTR': 0.1, 'FCRN-A': 0.22, 'FCRN-B': 0.06, 'baseline':0.},
        'ANAPATH': {'SRTC': 0., 'PRTR': 0., 'DRTR': 0., 'FCRN-A': 0.20, 'FCRN-B': 0.24},
        'GANGLIONS': {'SRTC': 0.68, 'PRTR': 0.42, 'DRTR': 0.32, 'FCRN-A': 0.26, 'FCRN-B': 0.22},
        'CRC': {'SRTC': 0., 'PRTR': 0., 'DRTR': 0., 'FCRN-A': 0.1, 'FCRN-B': 0.1}
    }

def model_assessment_latex(dataset, method, files):

    for f in files:
        if dataset in f and method in f:
            if 'assessment-assessment' in f:
                asmnt = pd.read_csv(f)
                print "ASSESSMENT"
                print tabulate(best_gs(asmnt, sigma=SIGMAS[dataset][method], kappa=KAPPAS[dataset][method]), tablefmt='latex')
            elif 'assessment' in f:
                gs = pd.read_csv(f)
                print tabulate(best_gs(gs), tablefmt='latex')
                # plot_pr_curve_by_sigma_assessment(gs, {'dataset':dataset, 'method':method, 'param':'assessment'})

    plt.show()
    # print tabulate(assessment(asmnt), tablefmt='latex')


def assessment_pr_curve(dataset, files):

    plt.figure()
    ff = [''] * 5
    for f in files:
        if dataset in f and ('assessment-assessment' in f):
            if 'SRTC' in f:
                ff[0] = f
            elif 'PRTR' in f:
                ff[1] = f
            elif 'DRTR' in f:
                ff[2] = f
            elif 'FCRN-A' in f:
                ff[3] = f
            elif 'FCRN-B' in f:
                ff[4] = f

    for f in ff:
        if dataset in f and ('assessment-assessment' in f):
            name = os.path.splitext(os.path.basename(f))[0]
            name_split = name.split('_')
            infos = dict()
            infos['dataset'] = name_split[0]
            infos['method'] = name_split[1]
            infos['param'] = '_'.join(name_split[2:])

            if infos['dataset'] == 'GANGLIONS' and infos['method'] == 'SRTC':
                t = 0.08
            else:
                t = 0.


            if infos['method'] == 'SRTC':
                c = 'C0'
            elif infos['method'] == 'PRTR':
                c = 'C1'
            elif infos['method'] == 'DRTR':
                c = 'C2'
            elif infos['method'] == 'FCRN-A':
                c = 'C3'
            elif infos['method'] == 'FCRN-B':
                c = 'C4'
            df = pd.read_csv(f)
            df = df[df['param_post_sigma'] == SIGMAS[infos['dataset']][infos['method']]]
            p, r, p_interpolate, r_interpolate, auc = pr_curve(df, t)
            plt.plot(r, p, '.-', color=c, label='{} (AUC={:.4f})'.format(infos['method'], auc))

    if dataset == 'ANAPATH':
        SRTC = {'mean_precision_score': [0.4962],
                'mean_recall_score': [0.4962]}
        p, r, p_interpolate, r_interpolate, auc = pr_curve(pd.DataFrame(SRTC,
                        columns=['mean_precision_score', 'mean_recall_score']), t=0.)
        plt.plot(r, p, '.-', color='C0', label='{}'.format('SRTC'))
        PRTR = {'mean_precision_score': [0.5312],
                'mean_recall_score': [0.5312]}
        p, r, p_interpolate, r_interpolate, auc = pr_curve(pd.DataFrame(PRTR,
                        columns=['mean_precision_score', 'mean_recall_score']), t=0.)
        plt.plot(r, p, '.-', color='C1', label='{}'.format('PRTR'))
        DRTR = {'mean_precision_score': [0.5207],
                'mean_recall_score': [0.5207]}
        p, r, p_interpolate, r_interpolate, auc = pr_curve(pd.DataFrame(DRTR,
                     columns=['mean_precision_score', 'mean_recall_score']), t=0.)
        plt.plot(r, p, '.-', color='C2', label='{}'.format('DRTR'))

    if dataset == 'CRC--':
        SRTC = {'mean_precision_score': [0.6329],
                'mean_recall_score': [0.6329]}
        p, r, p_interpolate, r_interpolate, auc = pr_curve(pd.DataFrame(SRTC,
                        columns=['mean_precision_score', 'mean_recall_score']), t=0.)
        plt.plot(r, p, '.-', color='C0', label='{}'.format('SRTC'))
        PRTR = {'mean_precision_score': [0.7365],
                'mean_recall_score': [0.7365]}
        p, r, p_interpolate, r_interpolate, auc = pr_curve(pd.DataFrame(PRTR,
                        columns=['mean_precision_score', 'mean_recall_score']), t=0.)
        plt.plot(r, p, '.-', color='C1', label='{}'.format('PRTR'))
        DRTR = {'mean_precision_score': [0.6853],
                'mean_recall_score': [0.6853]}
        p, r, p_interpolate, r_interpolate, auc = pr_curve(pd.DataFrame(DRTR,
                     columns=['mean_precision_score', 'mean_recall_score']), t=0.)
        plt.plot(r, p, '.-', color='C2', label='{}'.format('DRTR'))

        DRTR = {'mean_precision_score': [0.758],
                'mean_recall_score': [0.827]}
        p, r, p_interpolate, r_interpolate, auc = pr_curve(pd.DataFrame(DRTR,
                                                                        columns=['mean_precision_score',
                                                                                 'mean_recall_score']), t=0.)
        plt.plot(r, p, '.-', color='C5', label='{}'.format('SC-CNN (M=1)'))
        DRTR = {'mean_precision_score': [0.781],
                'mean_recall_score': [0.823]}
        p, r, p_interpolate, r_interpolate, auc = pr_curve(pd.DataFrame(DRTR,
                                                                        columns=['mean_precision_score',
                                                                                 'mean_recall_score']), t=0.)
        plt.plot(r, p, '.-', color='C6', label='{}'.format('SC-CNN (M=2)'))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0.,1.)
    plt.ylim(0.,1.)
    plt.title('{} - Test set PR Curves'.format(infos['dataset']))
    plt.legend()
    plt.savefig(os.path.join(FIGURES, dataset + '_pr_curve.pdf'))
    plt.show()

if __name__ == '__main__':
    path = '../../results/'
    onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for ds in ['ANAPATH','CRC']:
        assessment_pr_curve(ds, onlyfiles)
        for m in ['FCRN-A', 'FCRN-B', 'PRTR', 'SRTC', 'DRTR']:
            print ds
            print m
            model_assessment_latex(ds, m, onlyfiles)
            model_selection_latex(ds, m, onlyfiles)
            print os.linesep
    # for f in onlyfiles:
    #     if 'assessment' not in f:
    #         plots_from_dataframe(os.path.join(path, f))

    # plot_assessment_from_dataframe('../../results/ANAPATH_FCRN-A_assessment-assessment.csv')