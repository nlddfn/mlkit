import pprint
import logging
import random
import string

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from imblearn.metrics import sensitivity_specificity_support
from sklearn.utils import compute_sample_weight
from sklearn.metrics import (confusion_matrix,
                             f1_score,
                             balanced_accuracy_score,
                             accuracy_score,
                             median_absolute_error,)
pd.options.display.max_rows = 999


def eda(df, target=None, label=None, what=['basic']):
    """Produce EDA for target data."""
    # Choose what to see: basic, missingness, imbalance, target_agg,
    if 'basic' in what:
        print('\n****EDA for df****\n')
        print('Columns and head(2)\n')
        pprint.pprint(df.T.iloc[:, :2])
        print('\nBasic stats:\n')
        print((df.describe().T))
    if 'missing' in what:
        print('\nType and missingness\n')
        miss = pd.concat([df.dtypes, df.isnull().sum(axis=0), ], axis=1)
        miss.columns = ['Type', 'Missing']
        miss['Miss %'] = np.round(miss['Missing'] / df.shape[0] * 100)
        print(miss)
    if 'remarkable' in what:
        print(('\nNumber of outliers per feature out of %i samples\n' % df.shape[0]))
        quart = df.describe().transpose().drop(['count', 'mean', 'std'], axis=1)
        quart = quart.assign(Lval=np.maximum(quart['min'],
                             2.5 * quart['25%'] - 1.5 * quart['75%']),
                             Hval=np.minimum(quart['max'],
                             2.5 * quart['75%'] - 1.5 * quart['25%']))
        quart['index'] = quart.index
        quart['Outliers %'] = quart['index'].apply(
            lambda x: sum(
                (df[x] < quart.loc[x, 'Lval']) | (df[x] > quart.loc[x, 'Hval'])
            ) / float(df.shape[0]) * 100.0)
        quart.drop(['index', 'min', 'max', '25%', '75%'], inplace=True, axis=1)
        print(quart)

    # targets needed

    if isinstance(target, pd.core.series.Series):
        target = target.to_frame(label)
        label = lazyList(label)

    for item in label:
        tmp = pd.concat([df, target[item]], axis=1)

        if (item is not None) and ('target_agg' in what):
            print(('\nAggregation by target: %s\n' % (item)))
            print(np.round((tmp.groupby(by=item, axis=0).agg(np.median).T), 2))

        if (item is not None) and ('target_imbalance' in what):
            print(('\nClass imbalance for %s\n' % (item)))
            classes = np.unique(tmp[item].values)
            print((pd.DataFrame(confusion_matrix(tmp[item], tmp[item]),
                                columns=classes, index=classes)))


def sen_spe_sup(y_true, y_pred, labels, average=None):
    """Produce sensitivity, specificity, support."""
    sss = sensitivity_specificity_support(y_pred=y_pred, y_true=y_true,
                                          average=average)
    wht = ['Sensitivity', 'Specificity', 'Support']
    return pd.DataFrame(dict(list(zip(wht, sss))), index=labels).T


def label_encoder(base, y):
    """Produce label encoding."""
    # Set classes
    clas_dic = dict(list(zip(list(range(base.nclas)), base.classes)))
    return np.vectorize(clas_dic.get)(y)


def label_decoder(base, y):
    """Produce label decoding."""
    clas_dic = dict(list(zip(base.classes, list(range(base.nclas)))))
    return np.vectorize(clas_dic.get)(y)


def matthews_binary(y_true, y_pred):
    """Calculate Matthew correlation coefficient."""
    [[tn, fp], [fn, tp]] = coo_matrix(
        (np.ones((len(y_true))), (y_true, y_pred)), shape=(2, 2)).toarray().tolist()
    score = (tp * tn - fp * fn) / (
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
    return score


def base_scorer(y_true, y_pred, **params):
    score = params.get('score', accuracy_score)
    sample_weight = compute_sample_weight('balanced', y_true)
    return score(y_true, y_pred, sample_weight=sample_weight)


def ssm_scorer(y_true, y_pred, **params):
    score = params.get('score', accuracy_score)
    y_labeled = y_true[y_true != -1]
    y_pred_labeled = y_pred[y_true != -1]
    sample_weight = compute_sample_weight('balanced', y_labeled)
    return score(y_labeled, y_pred_labeled, sample_weight=sample_weight)


def lazyList(x):
    """Treat the case for multiple outputs."""
    if type(x) == list:
        return x
    else:
        return [x]


def check_test_class(y_true, y_pred, class_weight, classes):
    """Produce some basic QA when a benchamerk is requested."""

    print(('\nBalanced accuracy score: %.2f' % (
        balanced_accuracy_score(y_true, y_pred))))
    print(('\nF1 score: %.2f' % (f1_score(y_true, y_pred, average='weighted'))))
    try:
        print(('\nMatthews coefficient: %.2f' % (matthews_binary(y_true, y_pred))))
        print(('\nConfusion M for class {d[0]} and threshold = {d[1]}\n'.format(
              d=(list(class_weight.keys()), list(class_weight.values())))))
        print('\n')
        print((pd.DataFrame(confusion_matrix(y_pred=y_pred, y_true=y_true),
                            columns=classes, index=classes)))
    except ZeroDivisionError:
        print('No Conversion in the data')


def check_test_regr(df, label_true, label_pred):
    """Produce some basic QA when a benchamerk is requested."""
    print('Median absolute error for {label} is {score}'.format(
        label=label_true, score=median_absolute_error(df[label_true], df[label_pred])))


def setup_logger(log_file=None, level=logging.DEBUG):
    """Setup loggers"""
    # create logger
    log_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    ch.setFormatter(formatter)
    # add the handler to the logger
    logger.addHandler(ch)

    if log_file is not None:
        # create file handler which logs info messages
        fh = logging.FileHandler(filename=log_file, mode='a')

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)

    return logger
