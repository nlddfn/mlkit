import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.semi_supervised import LabelSpreading
from mlkit.BaseSupervised import BaseClassifier
from mlkit.BasePreProcessing import BasePreProcessing
import mlkit.base_model_utils as hp
from sklearn.metrics import (calinski_harabaz_score,
                             davies_bouldin_score)


class BaseUnsupervised(BasePreProcessing):
    """docstring for BaseUnsupervised."""

    def EDA(self, what=['basic']):
        # This requires dataframes!
        if self._verbose > 0:
            return hp.eda(self._dX, target=self._y, label=self._target,
                          what=what)

    def GridSearchUnsupervised(self, estimator, grid, scoring, fit_params={}):
        """Hyperparameter tuning, for Unsupervised methods."""
        for g in ParameterGrid(grid):
            estimator.set_params(**g)
            model = estimator.fit(self._X, **fit_params)
            score = scoring(X=self._X, labels=model.labels_)
            # save if best
            best_score = -999.0
            if score > best_score:
                best_score = score
                best_grid = g
                best_est = estimator

        score = scoring(X=self._X, labels=model.labels_)

        print("Best score = {score}".format(score=best_score))
        print("Grid:", best_grid)
        # Extra metrics, for comparison
        print('Cal-Hab score = {score}'.format(
            score=calinski_harabaz_score(self._X, model.labels_)))
        print('Davies-Bouldin score = {score}'.format(
            score=davies_bouldin_score(self._X, model.labels_)))

        return best_est

    def clustering(self,
                   estimator,
                   model_name='clustering.pkl',
                   grid=None,
                   fit_params={},
                   scoring=None,
                   owr=False):

        self.printv('\nApply {clas}:\n'.format(clas=model_name[:-4]), 0)
        model_path = os.path.join(self.model_path, model_name)

        if (not os.path.isfile(model_path)) or owr:

            best_est = self.GridSearchUnsupervised(
                estimator,
                grid,
                scoring=scoring)

            # Create new attributes to store threshold, classes, and data schema
            setattr(best_est, 'data_schema', self._X.columns.values)

            # clustering
            labels = best_est.labels_

            # Save model
            joblib.dump(best_est, model_path)
        else:
            # Prediction/metrics
            best_est = joblib.load(model_path)
            self.printv('Existing model loaded\n', 0)
            try:
                labels = best_est.predict(self._X)
            except AttributeError:
                labels = best_est.fit_predict(self._X)

        print('Clustering segmentation')
        print(pd.Series(best_est.labels_).groupby(by=best_est.labels_).count())

        # Define New labels
        self._y = pd.Series(labels.astype('int'), index=self._X.index)
        self._target = 'label'

        return best_est


class BaseSSM(BaseClassifier):
    """docstring for BaseSSM."""

    def __init__(self, df, target, model_path, verbose=0):
        super(BaseSSM, self).__init__(df, target, model_path, verbose)
        self.label_index = df[df[target] != -1].index
        self.unlabel_index = df[df[target] == -1].index
        self.pipe_steps = []

    def get_unlabeled(self):
        return self._X.loc[self.unlabel_index, :]

    def update_labeled_data(self, X_label, y_label):
        self.X_train = self.X_train.append(X_label, ignore_index=True)
        self.y_train = self.y_train.append(y_label, ignore_index=True)
        print('New train shape = {shape}'.format(shape=self.X_train.shape))

    def set_test_data(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def label_classifier(self,
                         model_name='model.pkl',
                         grid=None,
                         fit_params={},
                         cv_params={},
                         metric='accuracy',
                         owr=False):

        self.printv('\nApply Label Spreading:\n', 0)
        model_path = os.path.join(self.model_path, model_name)

        if (not os.path.isfile(model_path)) or owr:

            self.printv('\nTrain model\n', 0)

            estimator = LabelSpreading(kernel='rbf',
                                       gamma=.25,
                                       max_iter=10,
                                       alpha=0,
                                       n_jobs=-1)
            if grid is None:
                grid = {'n_neighbors': [3, 5, 7, 11],
                        'gamma': [1, 5, 10, 20, 50],
                        'kernel': ['rbf', 'knn'],
                        'alpha': [0.05, 0.1, 0.2, 0.5],
                        'max_iter': [10, 50, 100, 250]}

            grid_search = GridSearchCV(estimator, grid,
                                       scoring='accuracy',
                                       cv=5,
                                       n_jobs=-1,
                                       fit_params=fit_params)
            grid_search.set_params(**cv_params)
            grid_search.fit(self.X_train, self.y_train)

            if self._verbose > 0:
                print((grid_search.best_params_))
                print(("Best {method} score: {score} \n".format(
                    method=grid_search.scoring,
                    score=grid_search.best_score_)))
                print(("Training {method} score: {score}\n".format(
                    method=grid_search.scoring,
                    score=grid_search.score(self.X_train, self.y_train))))

            # Set best model
            best_est = grid_search.best_estimator_
            # Create new attributes to store threshold, classes, and data schema
            setattr(best_est, 'class_threshold', {0: 0.5})
            setattr(best_est, 'model_classes', np.array(self.classes))
            setattr(best_est, 'data_schema', self._X.columns.values)

            # Save model
            joblib.dump(best_est, model_path)

            if self._verbose > 0:
                print('\n==================')
                print('Show results train')
                print('==================\n')
                self.get_clas_results(best_est, (self.X_train[self.y_train != -1],
                                                 self.y_train[self.y_train != -1]))

                print('\n==================')
                print('Show results test')
                print('==================\n')
                self.get_clas_results(best_est, (self.X_test, self.y_test))

        else:
            # Prediction/metrics
            best_est = joblib.load(model_path)
            self.printv('Existing model loaded\n', 0)
            self.get_clas_results(best_est, (self.X_test, self.y_test))

        return best_est
