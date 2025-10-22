import os 
import sys 
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.constants import *
from src.logger import logging
from src.exception import MyEXception
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.proj1_data import Proj1Data

class DataIngestion:
    def __init__(self , data_ingestion_config = DataIngestionConfig()):
        """parameters : 
                      data_ingestion_config : configuration for data ingestion config"""
        try:
            self.data_ingestion_config  = data_ingestion_config
        except Exception as e:
            raise MyEXception(e,sys)
        
    
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Exporting data from mongo db")
            my_data = Proj1Data()
            dataframe = my_data.export_collection_as_dataframe(collection_name = self.data_ingestion_config.collection_name)
            logging.info(f"Shape of Dataframe : {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path , exist_ok=True)

            logging.info(f"Saving expoted data to feature store {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path , index = False , header = True)
            return dataframe
        
        except Exception as e:
            raise MyEXception(e,sys)
        
    def split_data_as_train_test(self , dataframe : DataFrame) -> None:
        """This method is usedf to split the data into train and test data"""
        try:
            train_data , test_data = train_test_split(dataframe , test_size= self.data_ingestion_config.train_test_split_ratio)
            logging.info(f"Sucessfully split the data into train and test")
            logging.info(f"Train Data Shape : {train_data.shape} and Test Data Shape : {test_data.shape}")
            dir_name = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_name , exist_ok=True)
            logging.info("Saving train and test data")
            train_data.to_csv(self.data_ingestion_config.training_file_path , index = False , header = True)
            test_data.to_csv(self.data_ingestion_config.testing_file_path , index = False , header = True)
            logging.info(f"Training adta saved at : [{self.data_ingestion_config.training_file_path}] and testing data at : [{self.data_ingestion_config.testing_file_path}]")

        except Exception as e:
            raise MyEXception(e,sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """This method is used to inititate the data ingestion
        Returns :
            DataIngestionArtifact : artifact of data aingestion component"""
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("exported data from mongo db")
            self.split_data_as_train_test(dataframe=dataframe)
            logging.info(f"Performed train test split")

            data_ingestion_arifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,
                                                           test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f"Data Ingestion Artifact : {DataIngestionArtifact}")

            return data_ingestion_arifact
        except Exception as e:
            return MyEXception(e,sys)
