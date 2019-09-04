import os
# import re
import pandas as pd
import numpy as np
# import logging
from scipy.stats import ks_2samp, kruskal, chi2_contingency

from sklearn.metrics import (confusion_matrix,
                             median_absolute_error,
                             precision_score,
                             recall_score,
                             balanced_accuracy_score,
                             f1_score,
                             matthews_corrcoef)

from mlkit.base_model_utils import (eda,
                                    lazyList,
                                    label_encoder,
                                    sen_spe_sup)


class Data(object):
    """Data model object."""
    def __init__(self, df, logger, target=None):
        """Init."""
        if set(lazyList(target)).issubset(df.columns):
            # Supervised scenario: training
            self._target = target
            self._y = df[target]
            self._X = df.drop(lazyList(target), axis=1)
        elif target is not None:
            # Supervised scenario: prediction
            self._target = target
            self._X = df.copy()
        else:
            # Unsupervised scenario
            self._X = df.copy()

        # Indexing columns by type
        self.cat_cols = list(self._X.select_dtypes(include=['object']).columns)
        self.num_cols = list(self._X.select_dtypes(include=[np.number]).columns)
        self.date_cols = list(self._X.select_dtypes(include=[np.datetime64]).columns)

        # Create a quick report
        type_X = pd.DataFrame(self._X.dtypes, columns=['type']).T
        self.data_description = self._X.describe(include='all').append(type_X)

        #  Initiate logging
        self.log = logger

    def EDA(self, what=['basic']):
        """Exploratory data analysis. Max(20000 samples)"""
        eda_frac = min(1, 20000 / self._X.shape[0])
        self.log.info(
            f'EDA data is based on {round(eda_frac * 100, 2)}% of the total data')
        return eda(self._X.sample(frac=eda_frac),
                   target=self._y,
                   label=self._target,
                   what=what)

    def check_schema(self, model, atr='_X'):
        """Check whether the data schema post processing is valid."""
        model_schema = model.data_schema
        data_schema = getattr(self, atr).columns.values
        if len(model_schema) != len(data_schema):
            missing = list(np.setdiff1d(model_schema, data_schema))
            raise ValueError(f'Check schema failed because {missing} is/are missing.')
        if np.any(model_schema != data_schema):
            self.log.critical('\nCHECK SCHEMA:')
            self.log.critical('Expected')
            self.log.critical(model_schema)
            self.log.critical('\nFound:')
            self.log.critical(data_schema)
            raise ValueError('Data schema is corrupted!')

    def get_clas_results(self, model, data, avg='macro'):

        data_X, data_y = data
        y_true = label_encoder(self, data_y)

        if self.nclas == 2:
            prob = (model.predict_proba(data_X)[:, 1] >= model.class_threshold[0]
                    ).astype(int)
            y_pred = label_encoder(self, prob)
        else:
            y_pred = model.predict(data_X)

        # Sensitivity, Specificity, Support: Training
        self.log.info(
            ('\nBalanced accuracy score: %.3f' % (
                balanced_accuracy_score(y_true, y_pred))))
        self.log.info(('\nF1 score: %.3f' % (f1_score(y_true, y_pred, average=avg))))

        self.log.info(
            ('\nPrec. score: %.3f' % (precision_score(y_true, y_pred, average=avg))))
        self.log.info(('\nRecall score: %.3f' % (
            recall_score(y_true, y_pred, average=avg))))

        self.log.info(('\nMatthews coefficient: %.3f' % (
            matthews_corrcoef(y_true, y_pred))))

        self.log.info(f'\nConfusion M for threshold '
                      f'= {round(model.class_threshold[0], 2)}\n')
        self.log.info((pd.DataFrame(confusion_matrix(y_pred=y_pred, y_true=y_true),
                                    columns=self.classes, index=self.classes)))

        self.log.info('\nSensitivity, Specificity, Support:\n')
        self.log.info((sen_spe_sup(y_pred=y_pred, y_true=y_true,
                                   labels=self.classes, average=None)))

    def get_regr_results(self, model, data):

        data_X, data_y = data

        # Test and summarize
        y_pred = model.predict(data_X)
        y_true = data_y.values

        # Account for multioutput
        if len(y_pred.shape) == 1:
            y_pred = np.expand_dims(y_pred, axis=-1)
            y_true = np.expand_dims(y_true, axis=-1)

        for ind in range(y_pred.shape[1]):
            self.log.info(
                f'Median absolute error for feature {ind}'
                f' is {round(median_absolute_error(y_true[:, ind], y_pred[:, ind]), 2)}')

    def get_prediction(self, model, target=None):
        """Retrieve prediction while enforcing schema"""
        if target is None:
            target = self._target
        if getattr(model, 'return_prob', False):
            y = model.predict_proba(self._X[model.data_schema])[:, 1]
        else:
            y = model.predict(self._X[model.data_schema])
        return pd.DataFrame(data=y, index=self._X.index, columns=lazyList(target))

    def stack(self, df):
        self._X = pd.concat([self._X, df], axis=1)


class Validator(object):
    """Check whether raw data at serving time is compatible with training data."""
    def __init__(self, data_obj, model_path, n_samples=1000):
        self.raw_sample = data_obj._X.sample(n=n_samples)
        self.n_samples = n_samples
        self.num_cols = data_obj.num_cols
        self.cat_cols = data_obj.cat_cols
        self.model_path = model_path

        # Create a quick report for training data
        type_X = pd.DataFrame(data_obj._X.dtypes, columns=['type']).T
        self.data_description = data_obj._X.describe(include='all').append(type_X)

    def run_validation(self, test):
        # 1- Check schema
        train_schema = self.raw_sample.columns
        test_schema = test._X.columns
        train_sample = self.raw_sample
        test_sample = test._X.sample(n=self.n_samples)

        if len(train_schema) != len(test_schema):
            missing = list(np.setdiff1d(train_schema, test_schema))
            raise ValueError(f'Check schema failed because {missing} is/are missing.')
        elif np.any(train_schema != test_schema):
            test.log.critical('\nCHECK SCHEMA:')
            test.log.critical('Expected')
            test.log.critical(train_schema)
            test.log.critical('\nFound:')
            test.log.critical(test_schema)
            raise ValueError('Data schema is corrupted!')
        else:
            test.log.info('\nSchema checked succesfully:\n')

        # 2- Categoricals: Check whether levels are missing in the training data
        if len(self.cat_cols) > 0:
            train_levels = self.data_description.loc['unique', self.cat_cols]
            test_levels = test.data_description.loc['unique', test.cat_cols]
            check_levels = (test_levels > train_levels)
            if np.any(check_levels):
                missing = list(np.setdiff1d(test_levels, train_levels))
                raise ValueError(f'Unknown levels found in columns {missing}.')

        # 3- Check data compatibility / print description
        # Test whether data sample comes from the same population
        arr = np.full(shape=(3, len(train_schema)), fill_value=np.nan)
        for i, col in enumerate(train_schema):
            if col in self.num_cols:
                # Perform 2 samples KS test on numerical features
                arr[0][i] = (ks_2samp(test_sample[col],
                                      train_sample[col]).pvalue > .05)
                # Perform 2 samples Kruskal Wallis test
                arr[1][i] = (kruskal(test_sample[col],
                                     train_sample[col]).pvalue > .05)
            elif col in self.cat_cols:
                # Perform chi2 test on categorical features
                tab = pd.crosstab(test_sample[col], train_sample[col])
                arr[2][i] = (chi2_contingency(tab)[1] < .05)

        # Create a EDA for data at serving time
        arr = pd.DataFrame(data=arr,
                           index=['KS-test', 'KW-test', 'Pe-test'],
                           columns=test_schema)
        validation = test.data_description.append([arr])
        validation.to_csv(
            os.path.join(self.model_path, 'validation_check.csv'), index=True)
