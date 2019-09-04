import pandas as pd
import os
from sklearn.externals import joblib

from mlkit.BaseData import Data
from mlkit.BasePreProcessing import BasePreProcessing
from mlkit.BaseSupervised import BaseClassifier, BaseRegressor
from mlkit.BaseUnsupervised import BaseUnsupervised


class Pipeline(object):

    def __init__(
        self,
        task_list,
        model_path,
        set_type=None,
        pipe_name='model_pip.pkl',
        auto_save=False
    ):

        self.task_list = task_list
        self.model_path = model_path
        self.set_type = set_type
        self.pipe_name = pipe_name
        self.auto_save = auto_save
        # check whether task_list are valid
        if any(list(map(lambda x: type(x) != dict, list(zip(*task_list))[1]))):
            raise TypeError('Parameters must have type dict')

    def save(self):
        # Save Pipeline for serving time
        joblib.dump(self, os.path.join(self.model_path, self.pipe_name))


class Preproc_Pipeline(Pipeline):
    """Run pipeline for preprocessing."""

    def run(self, data):
        """Run pipeline."""
        # check whether input is valid
        if not isinstance(data, Data):
            raise ValueError('Data must be an instance of Data class')

        model_obj = BasePreProcessing(obj=data,
                                      model_path=self.model_path,
                                      auto_save=self.auto_save)

        for i, (method, params) in enumerate(self.task_list):
            data.log.warning('\nCurrent Task: {task}\n'.format(task=method))
            model_bin = getattr(model_obj, method, None)(**params)
            # Model_bin is != None only for models
            if model_bin is not None:
                # Save the model
                self.task_list[i][1]['model_bin'] = model_bin
        return model_obj.data

    def predict(self, data):
        '''Run pipeline using saved models.'''
        # check whether input is valid
        if not isinstance(data, Data):
            raise ValueError('Data must be an instance of Data class')

        model_obj = BasePreProcessing(obj=data,
                                      model_path=self.model_path,
                                      auto_save=self.auto_save)
        # Execute tasks
        for method, params in self.task_list:
            data.log.warning('\nCurrent Task: {task}\n'.format(task=method))
            getattr(model_obj, method)(**params)
        return model_obj.data


class ModelPipeline(Pipeline):
    """Run pipeline ending with a model."""
    problem_dict = {'regr': BaseRegressor,
                    'clas': BaseClassifier,
                    'unsu': BaseUnsupervised}

    def run(self, data):
        """Run pipeline."""
        # check whether input is valid
        if not isinstance(data, Data):
            raise ValueError('Data must be an instance of Data class')

        # Set model object
        model_obj = self.problem_dict[self.set_type](obj=data,
                                                     model_path=self.model_path,
                                                     auto_save=self.auto_save)
        for i, (method, params) in enumerate(self.task_list):
            data.log.warning('\nCurrent Task: {task}\n'.format(task=method))
            model_bin = getattr(model_obj, method)(**params)
            # Model_bin is != None only for models
            if model_bin is not None:
                # Save the model
                self.task_list[i][1]['model_bin'] = model_bin
        return (model_obj.data, model_bin)

    def predict(self, data):
        '''Run pipeline using saved models.'''
        # check whether input is valid
        if not isinstance(data, Data):
            raise ValueError('Data must be an instance of Data class')

        model_obj = self.problem_dict[self.set_type](obj=data,
                                                     model_path=self.model_path,
                                                     auto_save=self.auto_save)
        # Execute tasks
        for method, params in self.task_list:
            data.log.warning('\nCurrent Task: {task}\n'.format(task=method))
            model_bin = getattr(model_obj, method)(**params)

        pred = data.get_prediction(model_bin)
        return pred


class Ensemble(object):
    """Create ensemble and train pipelines."""
    def __init__(self,
                 pipe_list,
                 task_list,
                 set_type,
                 model_path,
                 ensemble_name='ensemble.pkl'):
        self.pipe_list = pipe_list
        self.task_list = task_list
        self.set_type = set_type
        self.model_path = model_path
        self.ensemble_name = ensemble_name

    def save(self):
        # Save stack for serving time
        joblib.dump(self, os.path.join(self.model_path, self.ensemble_name))

    def run(self, data):
        """Run ensemble."""
        # check whether input is valid
        if not isinstance(data, Data):
            raise ValueError('Data must be an instance of Data class')

        # Set new input as concat of ensemble pipelines
        # NB: Assume the ensemble has the same target

        pred_lst = []
        # Create ensemble data obj
        for i, pipe in enumerate(self.pipe_list):
            data, model = pipe.run(data)
            pred = data.get_prediction(model)
            pred_lst.append(pred)
        ens_df = pd.concat(pred_lst, axis=1)
        # create fake column names and add target
        ens_df.columns = list(range(ens_df.shape[1]))
        ens_df[data._target] = data._y

        # Create Data obj for final estimator
        ens_data = Data(df=ens_df,
                        log_path=getattr(data, 'log'),
                        target=getattr(data, '_target'))
        setattr(ens_data, 'classes', data.classes)
        setattr(ens_data, 'nclas', data.nclas)

        # Create Pipeline
        self.ens_pip = ModelPipeline(task_list=self.task_list,
                                     model_path=self.model_path,
                                     set_type=self.set_type,
                                     pipe_name=self.ensemble_name)
        # now execute the model on the ensemble
        data, model = self.ens_pip.run(ens_data)
        return (data, model)

    def predict(self, data):
        '''Run pipeline using saved models.'''
        # check whether input is valid
        if not isinstance(data, Data):
            raise ValueError('Data must be an instance of Data class')

        # Initiate ensemble
        pred_lst = []
        # Create ensemble data obj
        for pipe in self.pipe_list:
            pred = pipe.predict(data)
            pred_lst.append(pred)

        ens_df = pd.concat(pred_lst, axis=1)
        ens_df.columns = list(range(ens_df.shape[1]))

        ens_data = Data(df=ens_df,
                        log_path=getattr(data, 'model_path'),
                        target=getattr(data, '_target'))
        pred = self.ens_pip.predict(ens_data)

        return pred


class Stack(object):
    """Create stack."""

    def __init__(self, pipe_list, model_path, stack_name='stack.pkl'):
        self.pipe_list = pipe_list
        self.model_path = model_path
        self.stack_name = stack_name

    def save(self):
        # Save stack for serving time
        joblib.dump(self, os.path.join(self.model_path, self.stack_name))

    def run(self, data_list, ):
        """Run stack and train pipelines."""
        # check whether input is valid
        if not all([isinstance(data, Data) for data in data_list]):
            raise ValueError('Data must be an instance of Data class')

        # Initiate stack
        pipe = self.pipe_list[0]
        data, model = pipe.run(data_list[0])
        pred = data.get_prediction(model)
        # Overwrite pipe to account (save) for the trained model
        self.pipe_list[0] = pipe

        # Loop across other models
        for i, pipe in enumerate(self.pipe_list[1:]):
            data_list[i + 1].stack(pred)
            data, model = pipe.run(data_list[i + 1])
            pred = data.get_prediction(model)

    def predict(self, data_list):
        '''Run stack using saved models.'''
        # check whether input is valid
        if not all([isinstance(data, Data) for data in data_list]):
            raise ValueError('Data must be an instance of Data class')

        # Initiate stack
        pipe = self.pipe_list[0]
        pred = pipe.predict(data_list[0])
        pred_list = [pred]

        # Loop across other models
        for i, pipe in enumerate(self.pipe_list[1:]):
            data_list[i + 1].stack(pred)
            pred = pipe.predict(data_list[i + 1])
            pred_list.append(pred)

        return pd.concat(pred_list, axis=1)
