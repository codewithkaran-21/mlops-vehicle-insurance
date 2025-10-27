import sys 
import os 

import pandas as pd
from pandas import DataFrame

from src.exception import MyEXception
from src.logger import logging
from sklearn.pipeline import Pipeline

class TargetValueMapping:
    def __init__(self):
        self.yes : int = 0
        self.no : int = 1

class MyModel:
    def __init__(self , preprocessing_object : Pipeline , trained_model_object : object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self , dataframe : DataFrame) -> DataFrame:
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        applies scaling using preprocessing_object, and performs prediction on transformed features.
        """
        try:
            logging.info("Starting prediction process")

            transformed_feature = self.preprocessing_object.transform(dataframe)
            logging.info("Using the trained model to get prediction")

            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions
    
        except Exception as e:
            raise MyEXception(e,sys)
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"