
from mlkit.BaseData import Data, Validator
from tests.test_base import TestBase


class TestData(TestBase):

    def test_basics(self, setup_model):
        # First introduce fake data and model
        obj = self.data_mock_clas
        assert type(obj) == Data, f'Expected type Data, found {type(obj)}'
        assert obj._target == 'target', 'Wrong target label detected'

        # Produce EDA for data
        obj.EDA(
            what=['basic', 'missing', 'remarkable', 'target_agg', 'target_imbalance']
        )

    def test_validation(self, setup_data):
        # First introduce fake data and model
        data = self.data_mock_clas

        v = Validator(
            data_obj=data,
            model_path='examples',
            n_samples=100
        )
        v.run_validation(test=data)

    def test_mock_models(self, setup_model):
        # Classification
        data_clas = self.data_mock_clas
        model = self.model_mock_clas
        model.set_classes([0, 1])
        estimator = model.base_dummy()

        data_clas.get_clas_results(
            model=estimator,
            data=(data_clas._X, data_clas._y)
        )

        # Regression
        data_regr = self.data_mock_regr
        model = self.model_mock_regr
        estimator = model.base_dummy()

        data_regr.get_regr_results(
            model=estimator,
            data=(data_regr._X, data_regr._y)
        )

        # Check getting the prediction from the fake model
        data_regr.get_prediction(model=estimator, target='target')

        # Check schema to the trained "fake" model
        data_regr.check_schema(estimator)
