import os
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from clver.model_utils.BasePreProcessing import BasePreProcessing
from clver.model_utils.base_model_utils import matthews_binary


class BaseClassifier(BasePreProcessing):
    """docstring for BaseClassifier."""

    def base_classifier(self,
                        estimator,
                        model_name='model.pkl',
                        grid=None,
                        fit_params={},
                        cv_params={},
                        metric=matthews_binary,
                        return_prob=False,
                        owr=False,
                        model_bin=None):

        self.log('\nApply {clas}:\n'.format(clas=model_name[:-4]), 'warning')
        model_path = os.path.join(self.model_path, model_name)
        self.log(f"*** MODEL_bin {model_bin}****", 'warning')

        if ((not os.path.isfile(model_path)) or owr) and (model_bin is None):
            self.log('\nTrain model\n')

            # Partition data
            self.partition_data(test_size=cv_params.pop('test_size'))
            # If early stop is set (eg. XGB), define eval_set in fit_params
            if 'early_stopping_rounds' in fit_params:
                fit_params['eval_set'] = [
                    (getattr(self.data, 'X_train'), getattr(self.data, 'y_train')),
                    (getattr(self.data, 'X_test'), getattr(self.data, 'y_test'))]

            # Define estimator and grid
            if (estimator is None) or (grid is None):
                self.log('No estimator or grid found: Set the default model')
                estimator = RandomForestClassifier(class_weight='balanced',
                                                   n_jobs=8,
                                                   criterion='gini',
                                                   verbose=0,
                                                   oob_score=True)

                grid = {"n_estimators": [50, 100, 200],
                        "min_samples_leaf": [4],
                        "min_samples_split": [2],
                        "max_depth": [4, 8, 12]}
            try:
                grid_search = GridSearchCV(estimator,
                                           grid,
                                           verbose=1,
                                           scoring='f1_macro',
                                           cv=5,
                                           n_jobs=8)
                grid_search.set_params(**cv_params)
                grid_search.fit(self.data.X_train, self.data.y_train, **fit_params)

                # logging
                self.log(grid_search.best_params_, 'warning')
                self.log("Best {method} score: {score} \n".format(
                    method=grid_search.scoring,
                    score=grid_search.best_score_), 'warning')
                self.log("Training {method} score: {score}\n".format(
                    method=grid_search.scoring,
                    score=grid_search.score(self.data.X_train, self.data.y_train)),
                    'warning')

                # Set best model
                best_est = grid_search.best_estimator_

            except ValueError:
                best_est = self.GridSearch(estimator,
                                           grid,
                                           scoring=cv_params.get('scoring', None))

            # Create new attributes to store threshold, classes, and data schema
            setattr(best_est, 'class_threshold', {0: .5})
            setattr(best_est, 'model_classes', np.array(self.data.classes))
            setattr(best_est, 'data_schema', self.data._X.columns.values)
            setattr(best_est, 'return_prob', return_prob)
            setattr(best_est, 'model_path', model_path)

            # If binary, use ROC to set model.class_threshold
            if self.data.nclas == 2:
                best_est.class_threshold = self.threshold_optimization_by_metric(
                    metric=metric, model=best_est)
                self.log(
                    'Classification threshold is {}'.format(
                        round(best_est.class_threshold[0], 2)))

            # Save model
            if self.auto_save:
                joblib.dump(best_est, model_path)

            # Logging
            self.log('\n==================')
            self.log('Show results train')
            self.log('==================\n')
            self.data.get_clas_results(best_est, (self.data.X_train, self.data.y_train))

            self.log('\n==================')
            self.log('Show results test')
            self.log('==================\n')
            self.data.get_clas_results(best_est, (self.data.X_test, self.data.y_test))

            # Feature_importance
            try:
                self.log('\nFeature importance:\n')
                feature_importance = list(zip(self.data._X.columns,
                                              best_est.feature_importances_))
                dtype = [('feature', 'S20'), ('importance', 'float')]
                feature_importance = np.array(feature_importance, dtype=dtype)
                feature_sort = np.sort(feature_importance,
                                       order='importance')[::-1]
                self.log(feature_sort)
            except AttributeError:
                pass

        elif os.path.isfile(model_path):
            # File exists/prediction
            best_est = joblib.load(model_path)
            self.log('Existing model loaded\n')

        else:
            # Prediction in pipeline
            best_est = model_bin
            self.log('Existing model loaded\n')
        return best_est


class BaseRegressor(BasePreProcessing):

    def base_regressor(self,
                       estimator,
                       model_name,
                       grid=None,
                       fit_params={},
                       cv_params={},
                       owr=False,
                       model_bin=None):

        self.log('\nApply regressor:\n')
        model_path = os.path.join(self.model_path, model_name)

        if ((not os.path.isfile(model_path)) or owr) and (model_bin is None):
            self.log('\nTrain model\n')

            # Partition data
            self.partition_data(test_size=cv_params.pop('test_size'))

            # Define estimator and grid
            if (estimator is None) or (grid is None):
                estimator = RandomForestRegressor(
                    n_jobs=-8, criterion='mse', verbose=0)

                grid = {"n_estimators": [50, 100, 200],
                        "min_samples_leaf": [4],
                        "min_samples_split": [2],
                        "max_depth": [4, 8, 12]}

            grid_search_forest = GridSearchCV(estimator,
                                              grid,
                                              verbose=1,
                                              scoring='neg_mean_absolute_error',
                                              cv=5,
                                              n_jobs=8)
            grid_search_forest.set_params(**cv_params)
            grid_search_forest.fit(self.data.X_train, self.data.y_train, **fit_params)

            self.log(grid_search_forest.best_params_, 'warning')
            self.log(("Best {method} score: {score} \n".format(
                method=grid_search_forest.scoring,
                score=grid_search_forest.best_score_)), 'warning')
            self.log(("Training {method} score: {score}\n".format(
                method=grid_search_forest.scoring,
                score=grid_search_forest.score(
                    self.data.X_train, self.data.y_train))), 'warning')

            # Set best model
            best_est = grid_search_forest.best_estimator_
            # Create new attributes to store threshold, classes, and data schema
            setattr(best_est, 'data_schema', self.data._X.columns.values)
            setattr(best_est, 'model_path', model_path)

            # Save model
            if self.auto_save:
                joblib.dump(best_est, model_path)

            # Logging
            self.log('\n==================')
            self.log('Show results train')
            self.log('==================\n')
            self.data.get_regr_results(best_est, (self.data.X_train, self.data.y_train))

            self.log('\n==================')
            self.log('Show results test')
            self.log('==================\n')
            self.data.get_regr_results(best_est, (self.data.X_test, self.data.y_test))

            try:
                self.log('\nFeature importance:\n')
                feature_importance = list(zip(self.data._X.columns,
                                              best_est.feature_importances_))
                dtype = [('feature', 'S20'), ('importance', 'float')]
                feature_importance = np.array(feature_importance, dtype=dtype)
                feature_sort = np.sort(
                    feature_importance, order='importance')[::-1]
                self.log(feature_sort)
            except AttributeError:
                pass

        elif os.path.isfile(model_path):
            # File exists/prediction
            best_est = joblib.load(model_path)
            self.log('Existing model loaded\n')

        else:
            # Prediction in pipeline
            best_est = model_bin
            self.log('Existing model loaded\n')

        return best_est
