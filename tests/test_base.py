import pytest
import logging
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from mlkit.BaseData import Data
from mlkit.BaseSupervised import BaseClassifier, BaseRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler()
    ])


class DummyClas(BaseClassifier):

    def base_dummy(self):
        model = DummyClassifier(strategy='uniform')
        setattr(model, 'class_threshold', {0: .5})
        setattr(model, 'model_classes', np.array(self.data.classes))
        setattr(model, 'data_schema', self.data._X.columns.values)
        setattr(model, 'return_prob', 0.5)
        setattr(model, 'model_path', 'model_path')
        return model.fit(self.data._X, self.data._y)


class DummyRegr(BaseRegressor):

    def base_dummy(self):
        model = DummyRegressor(strategy='mean')
        setattr(model, 'data_schema', self.data._X.columns.values)
        setattr(model, 'model_path', 'model_path')
        return model.fit(self.data._X, self.data._y)


class TestBase:
    """Create fake data and fake model."""
    N_ITEMS = 300
    N_FEAT = 20
    logger = logging.getLogger()

    @pytest.fixture
    def setup_raw_data(self, tmpdir):
        # Mock classifier
        self.df_mock_clas = self.get_randn_matrix(target='binary')
        # Mock regressor
        self.df_mock_regr = self.get_randn_matrix(target='regr')

    @pytest.fixture
    def setup_data(self, tmpdir):
        # Mock classifier
        self.data_mock_clas = self.get_data_obj(target='binary')
        # Mock regressor
        self.data_mock_regr = self.get_data_obj(target='regr')

    @pytest.fixture
    def setup_model(self, tmpdir):
        # Mock classifier
        self.data_mock_clas = self.get_data_obj(target='binary')
        self.model_mock_clas = self.get_dummy_model(
            data=self.data_mock_clas,
            type='clas'
        )
        # Mock regressor
        self.data_mock_regr = self.get_data_obj(target='regr')
        self.model_mock_regr = self.get_dummy_model(
            data=self.data_mock_regr,
            type='regr'
        )

    def get_data_obj(self, target):
        obj = Data(
            df=self.get_randn_matrix(target),
            logger=self.logger,
            target='target')
        return obj

    def get_randn_matrix(self, target=None):
        d = pd.DataFrame(
            data=np.random.randn(self.N_ITEMS, self.N_FEAT))
        if target == 'binary':
            d['target'] = np.random.choice([0, 1], self.N_ITEMS)
        elif target == 'regr':
            d['target'] = np.random.randn(self.N_ITEMS)
        return d

    def get_dummy_model(self, data, type):
        if type == 'clas':
            return DummyClas(
                obj=data,
                model_path='',
                verbose=True,
                auto_save=False
            )
        elif type == 'regr':
            return DummyRegr(
                obj=data,
                model_path='',
                verbose=True,
                auto_save=False
            )
        else:
            raise('Choose either clas or regr')
