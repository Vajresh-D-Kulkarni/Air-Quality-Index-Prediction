import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                so2:float,
                no2:float,
                rspm:float,
                spm:float):
        
        self.so2 = so2,
        self.no2 = no2,
        self.rspm = rspm,
        self.spm = spm
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'so2':[self.so2],
                'no2':[self.no2],
                'rspm':[self.rspm],
                'spm':[self.spm]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)


