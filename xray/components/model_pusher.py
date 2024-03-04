import os
import sys
import inspect
from xray.logger import logging
from xray.exception import CustomException
from xray.entity.config_entity import ModelPusherConfig
from xray.entity.artifact_entity import ModelPusherArtifact

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config

    def build_and_push_bento_image(self):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Started the {current_function_name} method of {self.__class__.__name__} class")

        try:
            logging.info("Building the bento from bentofile.yaml")

            os.system("bentoml build")

            logging.info("Built the bento from bentofile.yaml")

            logging.info("Creating docker image for bento")

            os.system(
                f"bentoml containerize {self.model_pusher_config.BENTOML_SERVICE_NAME}:latest -t {self.model_pusher_config.BENTOML_ECR_LINK}/{self.model_pusher_config.bentoml_ecr_image}:latest"
            )

            logging.info("Created docker image for bento")

            logging.info("Logging into ECR")

            os.system(
                f"aws ecr get-login-password --region {self.model_pusher_config.BENTOML_ECR_REGION} | docker login --username {self.model_pusher_config.BENTOML_ECR_USERNAME} --password-stdin {self.model_pusher_config.BENTOML_ECR_LINK}"
            )

            logging.info("Logged into ECR")

            logging.info("Pushing bento image to ECR")

            os.system(
                f"docker push {self.model_pusher_config.BENTOML_ECR_LINK}/{self.model_pusher_config.BENTOML_ECR_URI}:latest"
            )

            logging.info("Pushed bento image to ECR")

            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")

        except Exception as e:
            raise CustomException(e, sys) from e
        


    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher.

        Output      :   Model pusher artifact
        """
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Started the {current_function_name} method of {self.__class__.__name__} class")

        try:
            self.build_and_push_bento_image()

            model_pusher_artifact = ModelPusherArtifact(
                bentoml_model_name=self.model_pusher_config.bentoml_model_name,
                bentoml_service_name=self.model_pusher_config.bentoml_service_name,
            )

            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e