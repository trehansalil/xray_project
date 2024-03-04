import os
import sys
from typing import Tuple
import inspect
import joblib
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from xray.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from xray.entity.config_entity import DataTransformationConfig
from xray.exception import CustomException
from xray.logger import logging


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, 
                 data_ingestion_artifact: DataIngestionArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        
    def transforming_training_data(self) -> transforms.Compose:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")        
        try:

            train_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.ColorJitter(
                        **self.data_transformation_config.COLOR_JITTER_TRANSFORMS
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(
                        self.data_transformation_config.RANDOMROTATION
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.NORMALIZE_TRANSFORMS
                    ),
                ]
            )

            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")

            return train_transform

        except Exception as e:
            raise CustomException(e, sys) from e
    
    def transforming_testing_data(self) -> transforms.Compose:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")

        try:
            test_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.NORMALIZE_TRANSFORMS
                    ),
                ]
            )

            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")

            return test_transform

        except Exception as e:
            raise CustomException(e, sys) from e
    
    def data_loader(
        self, train_transform: transforms.Compose, test_transform: transforms.Compose
    ) -> Tuple[DataLoader, DataLoader]:
        
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")        
        try:
            train_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform=train_transform,
            )

            test_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.test_file_path),
                transform=test_transform,
            )

            logging.info("Created train data and test data paths")

            train_loader: DataLoader = DataLoader(
                train_data, **self.data_transformation_config.DATA_LOADER_PARAMS
            )

            test_loader: DataLoader = DataLoader(
                test_data, **self.data_transformation_config.DATA_LOADER_PARAMS
            )

            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
            return train_loader, test_loader

        except Exception as e:
            raise CustomException(e, sys) from e
  
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        
        try:
            train_transform: transforms.Compose = self.transforming_training_data()

            test_transform: transforms.Compose = self.transforming_testing_data()

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)

            joblib.dump(
                train_transform, self.data_transformation_config.TRAIN_TRANSFORMS_FILE
            )

            joblib.dump(
                test_transform, self.data_transformation_config.TEST_TRANSFORMS_FILE
            )

            train_loader, test_loader = self.data_loader(
                train_transform=train_transform, test_transform=test_transform
            )

            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                transformed_train_object=train_loader,
                transformed_test_object=test_loader,
                train_transform_file_path=self.data_transformation_config.TRAIN_TRANSFORMS_FILE,
                test_transform_file_path=self.data_transformation_config.TEST_TRANSFORMS_FILE,
            )
                        
            logging.info(f"Returing the DataTransformationArtifacts using {current_function_name} method of {self.__class__.__name__} class")
            
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e 