import os 
import sys
from src.logger import logging 
from src.exception import MyEXception
from src.entity.config_entity import DataValidationConfig
from src.constants import *
from src.entity.artifact_entity import DataValidationArtifact , DataIngestionArtifact
import pandas as pd
from pandas import DataFrame
from src.utils.main_utils import read_yaml 
import json

class DataValidation:
    def __init__(self , data_ingestion_artifact : DataIngestionArtifact , data_validation_config : DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_Config = read_yaml(file_path=SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise MyEXception(e,sys)
        
    def validate_number_of_columns(self , dataframe : DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self.schema_Config["columns"])
            logging.info(f"Is required column present [{status}]")
            return status
        
        except Exception as e:
            raise MyEXception(e,sys)
        
    def is_column_exist(self , df : DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            misssing_numerical_columns = []
            misssing_categorical_columns = []
            for column in self.schema_Config["numerical_columns"]:
                if column not in dataframe_columns:
                    misssing_numerical_columns.append1(column)
            
            if len(misssing_numerical_columns) > 0:
                logging.info(f"Missing Numerical Column : {misssing_numerical_columns}")

            for column in self.schema_Config["categorical_columns"]:
                if column not in dataframe_columns:
                    misssing_categorical_columns.append(column)
            
            if len(misssing_categorical_columns) > 0:
                logging.info(f"Missing Categorical Columns : {misssing_categorical_columns}")

            return False if len(misssing_categorical_columns) > 0 or len(misssing_numerical_columns) else True
        except Exception as e:
            raise MyEXception(e,sys)
    @staticmethod        
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise MyEXception(e,sys)
        
    def initiate_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            validation_error_message = ""
            logging.info("Starting data validation")
            train_df , test_df = (DataValidation.read_data(self.data_ingestion_artifact.train_file_path),
                                  DataValidation.read_data(self.data_ingestion_artifact.test_file_path))
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_message += f'Columns are missing in training Dataframe.'
            else:
                logging.info(f"All required columns are present in the dataset : {status}")

            status = self.validate_number_of_columns(dataframe = test_df)
            if not status:
                validation_error_message += f"Columns are missing in test Dataframe."

            else:
                logging.info(f"All columns are present in test dataset")

            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_message += f"Columns are missing in trining datframe"

            else:
                logging.info(f"All required columns are present in the training dataset : {status}")
            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_message += f"Columns are missing in test Dataframe"

            else:
                logging.info(f"All required columns are present in the dataset : {status}")

            validation_status = len(validation_error_message) ==0
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_message,
                validation_report_file_path = self.data_validation_config.validation_report_file_path)
            os.makedirs(os.path.dirname(self.data_validation_config.validation_report_file_path) , exist_ok=True)

            validation_report = {
                "validation_status" : validation_status,
                "message" : validation_error_message.strip()
            }

            with open(self.data_validation_config.validation_report_file_path , "w") as report_file:
                json.dump(validation_report , report_file , indent=4)

            logging.info(f"Data Validation Artifact created and save to JSON File")
            logging.info(f"Data Validation Artidfact : {DataValidationArtifact}")

            return data_validation_artifact

        except Exception as e:
            raise MyEXception(e,sys)