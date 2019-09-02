import os
import pandas as pd
import numpy as np
from category_encoders import OneHotEncoder

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib

from clver.model_utils.BaseModel import BaseModel


class BasePreProcessing(BaseModel):
    """Provide the basic methods to preprocess_data."""

    def transform_target(self,
                         multiOutLabel=None,
                         params={'log': False, 'nclas': None,
                                 'classes': None, 'q': True,
                                 'cut': None}):
        # Multioutput or not
        if multiOutLabel is not None:
            y = self.data._y[multiOutLabel]
        else:
            y = self.data._y

        # apply logarithmic scale to target
        if params.get('log', False):
            self.data._y = np.log(self.data._y)

        # transform continuous target into classes (binarization)
        if params.get('nclas', False):
            self.data.nclas = params['nclas']

            # if the classes are imbalanced, introduce different quantiles
            cut = params.get('cut', self.data.nclas)

            if params.get('q', True):
                # Calculate classes
                clas, bins = pd.qcut(y, cut, labels=False, retbins=True)
            else:
                # Calculate classes
                clas, bins = pd.cut(y, cut, labels=False, retbins=True)

            if multiOutLabel is not None:
                self.data._y[multiOutLabel] = clas.astype(int)
            else:
                self.data._y = clas.astype(int)

            self.data._df[self.data._target] = y

            # Define class labels
            self.data.classes = params.get(
                'classes',
                ['C' + str(x) for x in range(self.data.nclas)])

            # Logging
            self.log('\nTransform Target:\n', level='info')
            self.log('Create classes:', level='info')
            self.log((self.data.classes), level='info')
            self.log('\nBoundaries:', level='info')
            self.log(bins, level='info')

    def apply_random_imputation(self):
        """Adjusted random imputation with replacement."""
        # Define transform
        def random_imputation(x):
            out = np.copy(x)
            missing = np.isnan(x)
            rnd_samp = np.random.choice(x[np.logical_not(missing)], x[missing].shape[0])
            out[missing] = np.nanmean(x) + (np.mean(rnd_samp) - rnd_samp)
            return out

        # Logging
        self.log('\nApply random imputatation\n', 'info')
        self.data._X = self.data._X.apply(random_imputation, axis=0)

    def apply_PCA(self,
                  var_pct=.95,
                  model_name='PCA_model.pkl',
                  owr=False,
                  model_bin=None):
        """Apply PCA."""
        self.log('Apply PCA:', 'info')
        model_path = os.path.join(self.model_path, model_name)

        if ((not os.path.isfile(model_path)) or owr) and (model_bin is None):
            # Apply PCA retaining var_pct% of the variance
            model_bin = PCA(n_components=var_pct, svd_solver='full')
            model_bin.fit(self.data._X)
            print((model_bin.explained_variance_ratio_))
            # Save model
            if self.auto_save:
                joblib.dump(model_bin, model_path)

        elif os.path.isfile(model_path):
            # File exists => load the model
            model_bin = joblib.load(model_path)
            self.log('Existing model loaded\n')

        else:
            # Prediction in pipeline
            self.log('Existing model loaded\n')

        self.data._X = pd.DataFrame(model_bin.transform(self.data._X),
                                    index=self.data._X.index)
        return model_bin

    def apply_IF(self,
                 model_name='IF.pkl',
                 params=None,
                 owr=False,
                 model_bin=None):

        self.log('\nApply isolation forest on data to reduce outliers\n', 'info')
        model_path = os.path.join(self.model_path, model_name)

        if ((not os.path.isfile(model_path)) or owr) and (model_bin is None):
            self.log('\nTrain IF\n')
            if params is None:
                params = {"n_estimators": 500,
                          "max_samples": self.data._X.shape[0] // 10,
                          "contamination": .05,
                          "n_jobs": -1,
                          "random_state": None}

            IF = IsolationForest()
            IF.set_params(**params)
            model_bin = IF.fit(self.data._X)

            # Save model
            if self.auto_save:
                joblib.dump(model_bin, model_path)

        elif os.path.isfile(model_path):
            # Load the model
            model_bin = joblib.load(model_path)
            self.log('Existing model loaded\n')

        else:
            # Prediction in pipeline
            self.log('Existing model loaded\n')

        # Apply model
        is_inlier = model_bin.predict(self.data._X)
        self.log('Apply IF', 'info')
        self.log('removed {n} samples'.format(n=np.sum(1 - is_inlier) / 2.), 'info')
        self.data.X_outlier = self.data._X.loc[is_inlier == -1, :]
        self.data._X = self.data._X.loc[is_inlier == 1, :]

        if hasattr(self.data, '_y'):
            self.data._y = self.data._y.loc[is_inlier == 1]

        return model_bin

    def standardize_feat(self,
                         model_name='standardize_model.pkl',
                         cols=None,
                         owr=False,
                         model_bin=None):
        """Standardize features."""
        self.log('Apply Standardize features')

        model_path = os.path.join(self.model_path, model_name)
        if cols is None:
            cols = self.data.num_cols

        if (not os.path.isfile(model_path)) or owr:
            model_bin = StandardScaler()
            model_bin.fit(self.data._X[cols])
            self.data._X[cols] = model_bin.transform(self.data._X[cols])
            setattr(model_bin, 'data_schema', self.data._X.columns.values)
            # Save model
            if self.auto_save:
                joblib.dump(model_bin, model_path)

        elif os.path.isfile(model_path):
            # File exists/prediction
            model_bin = joblib.load(model_path)
            self.data._X[cols] = model_bin.transform(self.data._X[cols])
            self.data.check_schema(model_bin, '_X')
            self.log('Existing model loaded\n')

        else:
            # Prediction in pipeline
            self.data._X[cols] = model_bin.transform(self.data._X[cols])
            self.data.check_schema(model_bin, '_X')
            self.log('Existing model loaded\n')

        return model_bin

    def inverse_standardize_feat(self, model_name='standardize_model.pkl', cols=None):
        model_path = os.path.join(self.model_path, model_name)
        if cols is None:
            cols = self.data.num_cols

        model = joblib.load(model_path)
        self.data._X[cols] = model.inverse_transform(self.data._X[cols])
        self.data.check_schema(model, '_X')

    def categoricals(self,
                     model_name='onehot_model.pkl',
                     cols=None,
                     owr=False,
                     model_bin=None):
        """Onehot encoder on categoricals."""

        self.log('Apply onehot encoder on categorical')
        model_path = os.path.join(self.model_path, model_name)
        if cols is None:
            cols = self.data.cat_cols

        if ((not os.path.isfile(model_path)) or owr) and (model_bin is None):
            self.log('\nTrain model\n')
            model_bin = OneHotEncoder(
                cols=cols,
                use_cat_names=True,
                handle_unknown='error',
                drop_invariant=False,
                impute_missing=False)
            model_bin.fit(self.data._X)
            self.data._X = model_bin.transform(self.data._X)
            setattr(model_bin, 'data_schema', self.data._X.columns.values)

            # Save model
            if self.auto_save:
                joblib.dump(model_bin, model_path)

        elif os.path.isfile(model_path):
            # File exists/prediction:
            model_bin = joblib.load(model_path)
            self.data._X = model_bin.transform(self.data._X)
            self.data.check_schema(model_bin, '_X')

        else:
            # Prediction in pipeline
            self.data._X = model_bin.transform(self.data._X)
            self.data.check_schema(model_bin, '_X')

        return model_bin

    def resample_data(self, data=(), only_labeled=False,
                      frac=1, replace=False, random_state=None, output=False):

        # Define resampled index
        if len(data) == 2:
            if only_labeled:
                y = getattr(self.data, data[1])[self.data.label_index]
                X = getattr(self.data, data[0]).loc[self.data.label_index, :]
            else:
                y = getattr(self.data, data[1])
                X = getattr(self.data, data[0])

            resample_index = pd.DataFrame(
                y, columns=['label']).groupby(
                    by='label', group_keys=False).apply(
                        lambda x: x.sample(frac=frac,
                                           random_state=random_state,
                                           replace=replace)).index
            # Save original attribute
            setattr(self.data, data[0] + '_orig', getattr(self.data, data[0]))
            setattr(self.data, data[1] + '_orig', getattr(self.data, data[1]))
            # Resample
            setattr(self.data, data[0], X.loc[resample_index, :])
            setattr(self.data, data[1], y[resample_index])
            self.log('Resampled train has {n} samples'.format(n=len(resample_index)))

            if output:
                return (getattr(self, data[0]), getattr(self, data[1]))

        elif len(data) == 1:
            if only_labeled:
                X = getattr(self.data, data[0]).loc[self.data.label_index, :]
            else:
                X = getattr(self.data, data[0])

            resample_index = X.sample(
                frac=frac, random_state=random_state, replace=replace).index

            # Save original attribute
            setattr(self.data, data[0] + '_orig', getattr(self, data[0]))
            # Resample
            setattr(self.data, data[0], X.loc[resample_index, :])
            self.log('Resampled train has {n} samples'.format(n=len(resample_index)))

            if output:
                return getattr(self, data[0])
