import os
import sys

from src.logger import logging
from src.exception import customException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class dataIngestionConfig:
    train_data_path : str=os.path.join('artifacts', 'train.csv')
    test_data_path : str=os.path.join('artifacts', 'test.csv')
    raw_data_path : str=os.path.join('artifacts', 'raw.csv')

class dataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig()
    
    def initiateDataIngestion(self):
        logging.info("Data Ingestion Method Starts")
        
        try:
            df = pd.read_csv(os.path.join('notebooks/data','Book.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Raw data is created')

            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header = True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        
        except Exception as e:
            logging.info('Exception occured at Data Ingestion Stage')
            raise customException(e, sys)
