import os 
import sys
from src.constants import *
from src.utils.main_utils import *
from src.entity.config_entity import  ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact , ModelTrainerArtifact , ClassificatonMetricArtifact
from src.exception import MyEXception
from src.logger import logging
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , recall_score , precision_score , f1_score

class ModelTrainer:
    def __init__(self , data_transformation_artifact : DataTransformationArtifact , 
                 model_trainer_config : ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self , train : np.array , test : np.array) -> Tuple[object , object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            logging.info("Training RandomForestClassifier with specified parameters")
            x_train , y_train , x_test , y_test = train[:,:-1] , train[:,-1] , test[:,:-1] , test[:,-1]
            logging.info("train-test split done")
            model = RandomForestClassifier(
                n_estimators= self.model_trainer_config._n_estimators,
                min_samples_split= self.model_trainer_config._min_samples_split,
                min_samples_leaf= self.model_trainer_config._min_samples_leaf,
                max_depth= self.model_trainer_config._max_depth,
                criterion= self.model_trainer_config._criterion,
                random_state = self.model_trainer_config._random_state)
            logging.info("model training going on")
            model.fit(x_train , y_train)
            logging.info("model training done")
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test , y_pred)
            f1 = f1_score(y_test , y_pred)
            precision = precision_score(y_test , y_pred)
            recall = recall_score(y_test , y_pred)

            metric_artifact = ClassificatonMetricArtifact(f1_score=f1 , precision_Score=precision , recall_score=recall)
            return model , metric_artifact
        except Exception as e:
            raise MyEXception(e,sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        logging.info("Entered initiate_model_trainer of ModelTrainer Class")
        try:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Started Model Training")
            train_arr = load_numpy_array_data(file_path= self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path= self.data_transformation_artifact.transformed_test_file_path)

            logging.info("Train-Test data loaded")

            trained_model , metric_artifact = self.get_model_object_and_report(train=train_arr , test= test_arr)
            logging.info("Model object and artifact loaded")
            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)
            logging.info("preprocessing object loaded successfully")

            if accuracy_score(train_arr[:,-1] , trained_model.predict(train_arr[:,:-1])) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with accuracy above the base score")
                raise Exception("No model found with score above the base score")
            
            logging.info("Saving new model as this model accuracy is better then previous one")

            pass

        except Exception as e:
            raise MyEXception(e,sys)

