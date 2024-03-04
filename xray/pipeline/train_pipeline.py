import inspect
import sys
from xray.logger import logging
from xray.exception import CustomException
from xray.components.data_ingestion import DataIngestion
from xray.components.data_transformation import DataTransformation
from xray.components.model_trainer import ModelTrainer
from xray.components.model_evaluation import ModelEvaluation
from xray.components.model_pusher import ModelPusher
from xray.entity.config_entity import (DataIngestionConfig, 
                                       DataTransformationConfig, 
                                       ModelTrainerConfig, 
                                       ModelEvaluationConfig, 
                                       ModelPusherConfig)
from xray.entity.artifact_entity import (DataIngestionArtifact, 
                                         DataTransformationArtifact, 
                                         ModelTrainerArtifact,
                                         ModelEvaluationArtifact, 
                                         ModelPusherArtifact)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        # self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Getting the data from GCloud Storage bucket using the {current_function_name} method of {self.__class__.__name__} class")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Got the train and validation data from GCloud Storage using the {current_function_name} method of {self.__class__.__name__} class")
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e   
    
    # def start_data_validation(self) -> DataValidationArtifact:
    #     current_function_name = inspect.stack()[0][3]
    #     try:
    #         logging.info(f"Starting Validation of data using the {current_function_name} method of {self.__class__.__name__} class")
    #         data_validation = DataValidation(data_validation_config = self.data_validation_config)
    #         data_validation_artifact = data_validation.initiate_data_validation()
    #         logging.info(f"Data Validated using the {current_function_name} method of {self.__class__.__name__} class")
    #         return data_validation_artifact
    #     except Exception as e:
    #         raise CustomException(e, sys) from e       
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Starting Transformation of data using the {current_function_name} method of {self.__class__.__name__} class")
            
            data_transformation = DataTransformation(
                data_transformation_config = self.data_transformation_config, 
                data_ingestion_artifact = data_ingestion_artifact               
            )
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation done using the {current_function_name} method of {self.__class__.__name__} class")
            
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e    

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Starting Model Training using the {current_function_name} method of {self.__class__.__name__} class")
            
            model_trainer = ModelTrainer(
                data_transformation_artifact = data_transformation_artifact, 
                model_trainer_config = self.model_trainer_config
            )
            
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model Training done using the {current_function_name} method of {self.__class__.__name__} class")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e                    
        
    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact) -> ModelEvaluationArtifact:
        
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Starting Model Evaluation using the {current_function_name} method of {self.__class__.__name__} class")
            
        try:
            model_evaluation = ModelEvaluation(data_transformation_artifact = data_transformation_artifact,
                                                model_evaluation_config=self.model_evaluation_config,
                                                model_trainer_artifact=model_trainer_artifact)

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info(f"Model Evaluation done using the {current_function_name} method of {self.__class__.__name__} class")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
        
    

    def start_model_pusher(self,) -> ModelPusherArtifact:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Starting Model Evaluation using the {current_function_name} method of {self.__class__.__name__} class")
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
            )
            logging.info("Initiated the model pusher")
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            
            logging.info(f"Model Evaluation done using the {current_function_name} method of {self.__class__.__name__} class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e  
                     
    def run_pipeline(self):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Started the {current_function_name} method of {self.__class__.__name__} class")
        
        try:
            logging.info(f"Starting Ingestion using the {current_function_name} method of {self.__class__.__name__} class")
            data_ingestion_artifact = self.start_data_ingestion()

            # logging.info(f"Starting Validation using the {current_function_name} method of {self.__class__.__name__} class")            
            # data_validation_artifact = self.start_data_validation()
            
            # if (not data_validation_artifact.imbalance_data_valid) | (not data_validation_artifact.raw_data_valid):
            #     raise Exception("Data format is not valid")            
 
            logging.info(f"Starting Transformation using the {current_function_name} method of {self.__class__.__name__} class")            
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
            
            logging.info(f"Starting Model Training using the {current_function_name} method of {self.__class__.__name__} class")            
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            logging.info(f"Starting Model Evaluation using the {current_function_name} method of {self.__class__.__name__} class")            
            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,
                                                                    data_transformation_artifact=data_transformation_artifact
            )
            
            model_pusher_artifact = self.start_model_pusher()    
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
        except Exception as e:
            raise CustomException(e, sys) from e           
  