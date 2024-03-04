import sys

from xray.exception import CustomException
from xray.pipeline.train_pipeline import TrainPipeline


def start_training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    start_training()