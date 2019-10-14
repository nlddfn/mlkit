# import os
import pandas as pd
import numpy as np
from mlkit.BaseData import Data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

from sklearn.metrics import (roc_curve,
                             average_precision_score,
                             balanced_accuracy_score,
                             precision_recall_curve)


class BaseModel(object):
    """Provide the basic methods to build a model."""

    def __init__(self, obj, model_path, verbose=0, auto_save=True):
        """Init."""
        if isinstance(obj, Data):
            self.data = obj
        else:
            raise ValueError('obj must be an instance of Data class')
        self.model_path = model_path
        self._verbose = verbose
        self.auto_save = auto_save

    def printv(self, obj, level=0):
        if self._verbose > level:
            print(obj)

    def log(self, string, level='info'):
        getattr(self.data.log, level)(string)

    def get_target(self):
        return self.data._y

    def set_classes(self, labels):

        setattr(self.data, 'classes', labels)
        setattr(self.data, 'nclas', len(labels))

        self.log('\nCreated classes:')
        self.log(self.data.classes)

    def partition_data(self, test_size=0.2, random_state=None, return_index=False):

        if getattr(self.data, 'label_index', None) is None:
            # set max size for test set to 20,000 samples
            test_size = min(test_size, 20000 / self.data._X.shape[0])
            self.log('Test data is {perc}% of the total data'.format(
                perc=round(test_size * 100, 2)))
            try:
                self.log('WARNING: Stratified partitioning (classification)')
                X_train, X_test, y_train, y_test = train_test_split(
                    self.data._X, self.data._y, test_size=test_size,
                    random_state=random_state,
                    stratify=self.data._y)

            except (AttributeError, ValueError):
                self.log('WARNING: Not stratified partitioning (regression)')
                X_train, X_test, y_train, y_test = train_test_split(
                    self.data._X, self.data._y, test_size=test_size,
                    random_state=random_state)

        else:
            self.log('Manual stratified partitioning, due to SSM')
            # set max size for test set to 10,000 samples
            test_size = min(
                test_size, 10000 / self.data._X.loc[self.data.label_index, :].shape[0])
            self.log('Test data is {}% of the total data'.format(test_size * 100))
            # Reshape target
            dtar = pd.DataFrame(self.data._y, columns=['label'])
            # Select random stratified fraction of labeled data for test
            test_index = dtar.loc[self.data.label_index].groupby(
                by='label', group_keys=False).apply(
                    lambda x: x.sample(frac=test_size, random_state=random_state)).index
            train_index = self.data.label_index.difference(test_index)
            X_test = self.data._X.loc[test_index, :]
            y_test = self.data._y[test_index]
            X_train = self.data._X.loc[train_index, :]
            y_train = self.data._y[train_index]
            self.log('Class imbalance after stratified partitioning:\n{count}'.format(
                count=y_train.value_counts()
            ))

        # save methods in data model
        setattr(self.data, 'X_train', X_train)
        setattr(self.data, 'y_train', y_train)
        setattr(self.data, 'X_test', X_test)
        setattr(self.data, 'y_test', y_test)

        if return_index:
            return (y_train.index, y_test.index)

    def GridSearch(self, estimator, grid, scoring='f1_macro'):
        """Hyperparameter tuning, without CV."""
        for g in ParameterGrid(grid):
            estimator.set_params(**g)
            estimator.fit(self.data.X_train, self.data.y_train)
            try:
                score = scoring(estimator=estimator,
                                X=self.data.X_test,
                                y_true=self.data.y_test)
            except TypeError:
                score = scoring(
                    y_pred=estimator.predict(self.data.X_test), y_true=self.data.y_test)
            # save if best
            best_score = -2
            if score > best_score:
                best_score = score
                best_grid = g
                best_est = estimator
        try:
            score = scoring(
                estimator=estimator, X=self.data.X_train, y_true=self.data.y_train)
        except TypeError:
            score = scoring(
                y_pred=estimator.predict(self.data.X_train), y_true=self.data.y_train)
        self.log("Best score = {score}".format(score=best_score), 'warning')
        self.log("Training score = {score}".format(score=score), 'warning')
        self.log("Grid:", best_grid)
        return best_est

    def threshold_optimization_by_metric(self, metric, model):
        """Calculate the best threshold for decision function."""
        if metric == 'ROC_AUC':
            return self.ROC_AUC(model)
        elif metric == 'Precision_Recall':
            return self.Precision_Recall(model)
        elif type(metric) == float:
            assert 0 < metric < 1., \
                'If the value for metric is numerical, it should be between 0 and 1'
            return {0: metric}
        else:
            score_lst = []
            thr_lst = np.linspace(.01, .99, 50, endpoint=True)
            prob = model.predict_proba(self.data.X_train)[:, 1]
            for i in thr_lst:
                y_pred = (prob > i).astype(int)
                try:
                    score_lst.append(metric(self.data.y_train, y_pred))
                except (ZeroDivisionError, ValueError):
                    score_lst.append(0)
            return {0: thr_lst[np.argmax(score_lst)]}

    def ROC_AUC(self, model):
        """Compute ROC for training."""
        fpr, tpr, thr = roc_curve(
            y_true=self.data.y_train,
            y_score=model.predict_proba(
                self.data.X_train)[:, 1])

        # Compute optimal threshold per class
        opt_thr = {0: thr[np.argmin(((1 - tpr) ** 2 + fpr ** 2) ** .5)]}
        return opt_thr

    def Precision_Recall(self, model):

        # Compute Precision-Recall and plot curve
        y_true = self.data.y_train.values
        y_pred = model.predict_proba(self.data.X_train)[:, 1]
        precision, recall, thresholds = precision_recall_curve(
            y_true=y_true, probas_pred=y_pred)

        # Best Threshold
        f1 = 2 * precision * recall / (precision + recall)
        best_threshold = thresholds[np.argmax(f1)]
        best_precision = precision[np.argmax(f1)]
        best_recall = recall[np.argmax(f1)]
        best_f1 = max(f1)

        prec_rec_auc = average_precision_score(y_true, y_pred, average='weighted')

        prec_rec_auc_test = average_precision_score(
            self.data.y_test.values,
            model.predict_proba(self.data.X_test)[:, 1], average='weighted')

        self.log(('The Precision-Recall AUC for training is: %.5f\n' % (prec_rec_auc)))
        self.log(('The Precision-Recall AUC for test is: %.5f\n' % (prec_rec_auc_test)))
        self.log(("Best f1={f1} with threshold={thr}, precision={pr}, recal={rec}\
            ".format(
            f1=best_f1, thr=best_threshold, pr=best_precision, rec=best_recall)))
        self.log(("Balanced accuracy for Training: {score}".format(
            score=balanced_accuracy_score(y_true, y_pred > best_threshold))))
