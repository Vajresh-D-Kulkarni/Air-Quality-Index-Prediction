import numpy as np
import pandas as pd
import sys, os

from src.logger import logging
from src.exception import CustomException

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object

from dataclasses import dataclass
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')

            logging.info('Data Transformation Pipeline Initiated')

            numerical_columns = ['SO2', 'NOx', 'RSPM', 'SPM']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scalar', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', num_pipeline, numerical_columns)
                ]
            )
            
            logging.info('Data Transformation Completed')
            
            return preprocessor
        
        except Exception as e:
            logging.info('Exception occured in Data Transformation')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Reading of train and test data completed')
            logging.info(f'Train Dataframe : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_tranformation_object()

            target_columns = 'AQI'
            drop_columns = [target_columns, '']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_columns]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_columns]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object = (
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            logging.info("Applied Preprocessed")
            
        except Exception as e:
            raise CustomException(e, sys)