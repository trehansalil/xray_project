import inspect
import sys
from typing import Tuple
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from xray.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelEvaluationConfig
from xray.exception import CustomException
from xray.logger import logging
from xray.ml.model import Net

class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_evaluation_config = model_evaluation_config

        self.model_trainer_artifact = model_trainer_artifact

    def configuration(self) -> Tuple[DataLoader, Module, float, Optimizer]:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class") 

        try:
            test_dataloader: DataLoader = (
                self.data_transformation_artifact.transformed_test_object
            )

            model: Module = Net()

            model: Module = torch.load(self.model_trainer_artifact.trained_model_path)

            model.to(self.model_evaluation_config.DEVICE)

            cost: Module = CrossEntropyLoss()

            '''optimizer: Optimizer = SGD(
                model.parameters(), **self.model_evaluation_config.OPTIMIZER_PARAMS
            )'''

            model.eval()

            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class") 

            return test_dataloader, model, cost

        except Exception as e:
            raise CustomException(e, sys) from e

    def test_net(self) -> float:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class") 

        try:
            test_dataloader, net, cost = self.configuration()

            with torch.no_grad():
                holder = []

                for _, data in enumerate(test_dataloader):
                    images = data[0].to(self.model_evaluation_config.DEVICE)

                    labels = data[1].to(self.model_evaluation_config.DEVICE)

                    output = net(images)

                    loss = cost(output, labels)

                    predictions = torch.argmax(output, 1)

                    for i in zip(images, labels, predictions):
                        h = list(i)

                        holder.append(h)

                    logging.info(
                        f"Actual_Labels : {labels}     Predictions : {predictions}     labels : {loss.item():.4f}"
                    )

                    self.model_evaluation_config.TEST_LOSS += loss.item()

                    self.model_evaluation_config.TEST_ACCURACY += (
                        (predictions == labels).sum().item()
                    )

                    self.model_evaluation_config.TOTAL_BATCH += 1

                    self.model_evaluation_config.TOTAL += labels.size(0)

                    logging.info(
                        f"Model  -->   Loss : {self.model_evaluation_config.TEST_LOSS/ self.model_evaluation_config.TOTAL_BATCH} Accuracy : {(self.model_evaluation_config.TEST_ACCURACY / self.model_evaluation_config.TOTAL) * 100} %"
                    )

            accuracy = (
                self.model_evaluation_config.TEST_ACCURACY
                / self.model_evaluation_config.TOTAL
            ) * 100

            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class") 

            return accuracy

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class") 

        try:
            accuracy = self.test_net()

            model_evaluation_artifact: ModelEvaluationArtifact = (
                ModelEvaluationArtifact(model_accuracy=accuracy)
            )

            logging.info("Returning the ModelEvaluationArtifacts")
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")     

            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e