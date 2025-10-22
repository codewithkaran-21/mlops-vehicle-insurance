import sys 
import pandas as pd 
import numpy as np 
from typing import Optional

from src.logger import logging
from src.exception import MyEXception
from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME

class Proj1Data:
    """A class used to export records as a Dataframe from mongo DB Database"""
    def __init__(self) -> None:
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyEXception(e,sys)
        
    def export_collection_as_dataframe(self , collection_name : str , databse_name : Optional[str] = None) -> pd.DataFrame:
        """Exports the entire records of the collection as a pandas Dataframe"""
        try:
            if databse_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[databse_name][collection_name]

            print("Fetching data from mongoDB")
            logging.info("Fetching data from mongoDB")
            df = pd.DataFrame(list(collection.find()))
            print(f"Data fetched with len : {len(df)}")
            logging.info(f"Data fetched with len : {len(df)}")
            if "id" in df.columns.tolist():
                df = df.drop(columns=["id"] , axis =1)
            df.replace({"na" : np.nan} , inplace= True)
            return df
        
        except Exception as e:
            raise MyEXception(e,sys)
            
            
