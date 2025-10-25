import os 
import sys 

import numpy as np 
import dill
import yaml 
from pandas import DataFrame
from src.logger import logging
from src.exception import MyEXception

def read_yaml(file_path : str) -> dict:
    try:
        with open(file_path , 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MyEXception(e,sys)
    
def write_yaml_files(file_path  :str , content : object , reaplace : bool = False) -> None:
    try:
        if reaplace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path) , exist_ok=True)
        with open(file_path , 'w') as file:
            yaml.dump(content , file)
    except Exception as e:
        raise MyEXception(e,sys)
    
def load_object(file_path : str) -> object:
    """
    Returns model/object from project directory.
    file_path: str location of file to load
    return: Model/Obj
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise MyEXception(e, sys) from e

def save_numpy_array_data(file_path : str , array : np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """

    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path , exist_ok=True)
    with open(file_path , 'wb') as file_obj:
        np.save(file_obj , array)

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise MyEXception(e, sys) from e

def save_object(file_path : str , obj : object):
    logging.info("Entered the save method object of utils")
    try:
        os.makedirs(os.path.dirname(file_path) , exist_ok= True)
        with open(file_path , "wb") as file_obj : 
            dill.dump(obj , file_obj)
        
        logging.info("Exited from the save method object of utils")

    except Exception as e:
        raise MyEXception(e,sys)
