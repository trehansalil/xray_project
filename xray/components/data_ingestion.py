import os
import sys
import inspect
from zipfile import ZipFile
from xray.logger import logging
from xray.exception import CustomException
from xray.configuration.cloud_storage import S3Operation
from xray.entity.config_entity import DataIngestionConfig
from xray.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.s3 = S3Operation()
        
    def get_data_from_s3(self) -> None:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")        
        try:
            self.s3.sync_folder_from_s3(
                folder=self.data_ingestion_config.data_path,
                bucket_name=self.data_ingestion_config.bucket_name,
                bucket_folder_name=self.data_ingestion_config.s3_data_folder,
            )

            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def unzip_and_clean(self):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH) as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
            
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
            return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")

        try:
            self.get_data_from_s3()

            data_ingestion_artifact: DataIngestionArtifacts = DataIngestionArtifacts(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path,
            )

            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e