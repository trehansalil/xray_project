import os
import io
import sys
import keras
import pickle
import inspect
from PIL import Image
from xray.logger import logging
from xray.constants import *
from xray.exception import CustomException
from xray.configuration.cloud_storage import S3Operation
from xray.components.data_transformation import DataTransformation
from xray.entity.config_entity import DataTransformationConfig
from xray.entity.artifact_entity import DataIngestionArtifact


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.aws_cloud = S3Operation()
        self.data_transformation = DataTransformation(data_transformation_config= DataTransformationConfig,
                                                      data_ingestion_artifacts=DataIngestionArtifact)


    
    def get_model_from_gcloud(self) -> str:
        """
        Method Name :   get_model_from_gcloud
        Description :   This method to get best model from google cloud storage
        Output      :   best_model_path
        """
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")  
        try:
            # Loading the best model from s3 bucket
            os.makedirs(self.model_path, exist_ok=True)
            self.aws_cloud.sync_folder_from_s3(self.bucket_name, self.model_name, self.model_path)
            best_model_path = os.path.join(self.model_path, self.model_name)
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e
        

    
    def predict(self,best_model_path,text):
        """load image, returns cuda tensor"""
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class") 
        try:
            if not os.path.exists(best_model_path):
                print(best_model_path)
                logging.info(f"Fetching the model from GCloud using the {current_function_name} method of {self.__class__.__name__} class") 
                best_model_path:str = self.get_model_from_gcloud()
            load_model=keras.models.load_model(best_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            
            text=self.data_transformation.data_cleaning(text)
            text = [text]            
            print(text)
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)
            print(seq)
            pred = load_model.predict(padded)
            pred
            print("pred", pred)
            if pred>0.5:

                print("hate and abusive")
                logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
                return "hate and abusive"
            else:
                print("no hate")
                logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
                return "no hate"
        except Exception as e:
            raise CustomException(e, sys) from e

    
    def run_pipeline(self,text):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class") 
        try:
            best_model_path = os.path.join(self.model_path, MODEL_NAME)
            
            if not os.path.exists(best_model_path):
                print(best_model_path)
                best_model_path: str = self.get_model_from_gcloud() 
            predicted_text = self.predict(best_model_path,text)
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e