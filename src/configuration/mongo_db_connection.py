import os 
import sys 
import pymongo
import certifi

from src.exception import MyEXception
from src.logger import logging
from src.constants import DATABASE_NAME , MONGODB_URL_KEY

# Load the certificate authority file to avoid timeout errors when connecting to MongoDB
ca = certifi.where()  

class MongoDBClient:
    """Mongo DB Client is responsible for establishing connection to the Mongo DB Database"""
    client = None

    def __init__(self , database_name : str = DATABASE_NAME):
        """Intializes connection to the Mongo DB Database if no existing connection found establishes a new one
        
        Parameters :
            database_name (str) : Name of the database to connect to
        """
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment variable {MONGODB_URL_KEY} is not set")
                # Establish a new MongoDB client connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url , tlsCAFile = ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info(f"Mongo DB Connection established successfully")
        except Exception as e:
            raise MyEXception(e ,sys)