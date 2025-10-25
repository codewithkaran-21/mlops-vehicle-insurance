import os 
import sys
import numpy as np 
import pandas as pd 
from pandas import DataFrame
from src.logger import logging
from src.exception import MyEXception
from src.constants import *
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact , DataIngestionArtifact , DataValidationArtifact
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml

class DataTransformation:
    def __init__(self , data_ingestion_artifact : DataIngestionArtifact ,
                 data_transformation_config : DataTransformationConfig , data_validation_artifact : DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyEXception(e,sys)
        
    @staticmethod
    def read_data(file_path : str) -> pd.DataFrame:
        try:
             return pd.read_csv(file_path)

        except Exception as e:
            raise MyEXception(e,sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.

        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()

            logging.info("Transformer Initialized : MinMaxSacler , StandardScaler")
            num_features = self._schema_config["num_features"]
            mm_columns= self._schema_config["mm_columns"]
            logging.info("Columns loadded successfully")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler" , numeric_transformer , num_features),
                    ("MinMaxScaler" , min_max_scaler , mm_columns)
                ],
                remainder="passthrough"
            )

            final_pipeline = Pipeline(steps=[("Preprocessor" ,preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            raise MyEXception(e,sys)
    
    def _map_gender_column(self , df):
        logging.info("Mapping gender column to binary numkbers")

        df['Gender'] = df['Gender'].map({"Female": 0,"Male" : 1}).astype(int)
        return df
    
    def _create_dumy_cloumns(self , df):
        """The method is used to create dummmy variables for categorical features"""

        logging.info("Creating dummy varibales for categorical features")

        df = pd.get_dummies(df , drop_first=True)
        return df
    
    def _rename_columns(Slef ,df):
        logging.info("Reamining specific columns and creating it to int")
        df = df.rename(columns = {
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year" , "Vehicle_Age_gt_2_Years" , "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')

        return df
    
    def _drop_id_column(self,df):
        """THE METHOD IS USED DO DROP THE ID COLUMN FROM THE DATABASE"""
        logging.info("Dropping id column from dataset")

        drop_col = self._schema_config['drop_columns']

        if drop_col in df.columns:
            df = df.drop(drop_col , axis =1)
        return df
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the Data Transformation component pipeline
        
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            train_df = self.read_data(file_path=self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            logging.info("Train-Test Data Loader")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN] ,axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Input and target cols defined for both train and test data")
            input_feature_train_df = self._map_gender_column(input_feature_train_df)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._create_dumy_cloumns(input_feature_train_df)
            input_feature_train_df = self._rename_columns(input_feature_train_df)

            input_feature_test_df = self._map_gender_column(input_feature_test_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._create_dumy_cloumns(input_feature_test_df)
            input_feature_test_df = self._rename_columns(input_feature_test_df)


            logging.info("Custom Transformation applied to train and test data")

            logging.info("Starting Data Transformation")

            preprocessor = self.get_data_transformer_object()
            logging.info("Got the processor object")
            logging.info("Initializing transformation for Training-Data")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Test-Data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end-to-end to train-test data")
            logging.info("APPLYING smooten for handling imbalanced data")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final , target_feature_train_final = smt.fit_resample(
                input_feature_train_arr , target_feature_train_df
            )

            input_feature_test_final , target_feature_test_final = smt.fit_resample(
                input_feature_test_arr , target_feature_test_df
            )

            logging.info("SMOOTEN applied to train-test df")

            train_arr = np.c_[input_feature_train_final , np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final , np.array(target_feature_test_final)]

            logging.info("Feature target concatenation done train-test df")
            save_object(self.data_transformation_config.transformed_object_file_path , preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path , array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path , test_arr)
            logging.info("Saving transformation object and transformed files")
            logging.info("Data Transformation completed successfully")

            return DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
        )

        except Exception as e:
            raise MyEXception(e,sys)



