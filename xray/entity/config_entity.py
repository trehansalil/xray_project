from dataclasses import dataclass
from xray.constants import *
import os
from torch import device

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.S3_DATA_FOLDER: str = S3_DATA_FOLDER
        self.BUCKET_NAME: str = BUCKET_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, S3_DATA_FOLDER)
        self.TRAIN_DATA_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, TRAIN_DATA_DIR)
        self.TEST_DATA_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, TEST_DATA_DIR)

# @dataclass        
# class DataValidationConfig:
#     def __init__(self):        
#         self.IMBALANCE_DATA_DIR = IMBALANCE_DATA_DIR
#         self.RAW_DATA_DIR = RAW_DATA_DIR
#         self.IMBALANCE_DATA_COLUMNS = IMBALANCE_DATA_COLUMNS
#         self.RAW_DATA_COLUMNS = RAW_DATA_COLUMNS
        
@dataclass        
class DataTransformationConfig:
    def __init__(self):    
        self.ARTIFACTS_DIR = ARTIFACTS_DIR
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_TRANSFORMS_FILE: str = os.path.join(
            self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TRAIN_TRANSFORMS_FILE
        )

        self.TEST_TRANSFORMS_FILE: str = os.path.join(
            self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TEST_TRANSFORMS_FILE
        )   

        self.COLOR_JITTER_TRANSFORMS: dict = {
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "saturation": SATURATION,
            "hue": HUE,
        }
        
        self.RESIZE: int = RESIZE
        self.CENTERCROP: int = CENTERCROP
        self.RANDOMROTATION: int = RANDOMROTATION
        self.NORMALIZE_TRANSFORMS: dict = {
            "mean": NORMALIZE_LIST_1,
            "std": NORMALIZE_LIST_2,
        }
        self.DATA_LOADER_PARAMS: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": SHUFFLE,
            "pin_memory": PIN_MEMORY,
        }    
        
@dataclass        
class ModelTrainerConfig:
    def __init__(self):    
        self.ARTIFACTS_DIR = ARTIFACTS_DIR    
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_BENTOML_MODEL_NAME = BENTOML_MODEL_NAME
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)  

        self.TRAIN_TRANSFORMS_KEY: str = TRAIN_TRANSFORMS_KEY
 
        self.RANDOM_STATE = RANDOM_STATE   
        self.EPOCH = EPOCH
        self.OPTIMIZER_PARAMS: dict = {"lr": LEARNING_RATE, "momentum": MOMENTUM_BETA}

        self.SCHEDULER_PARAMS: dict = {"step_size": STEP_SIZE, "gamma": GAMMA}

        self.DEVICE: device = DEVICE 
        
@dataclass
class ModelEvaluationConfig: 
    def __init__(self):
        self.ARTIFACTS_DIR = ARTIFACTS_DIR
        self.DEVICE: device = DEVICE
        self.TEST_LOSS: int = 0
        self.TEST_ACCURACY: int = 0
        self.TOTAL: int = 0
        self.TOTAL_BATCH: int = 0
        self.OPTIMIZER_PARAMS: dict = {"lr": LEARNING_RATE, "momentum": MOMENTUM_BETA}

@dataclass
class ModelPusherConfig:

    def __init__(self):
        self.BENTOML_MODEL_NAME: str = BENTOML_MODEL_NAME
        self.BENTOML_SERVICE_NAME: str = BENTOML_SERVICE_NAME
        self.TRAIN_TRANSFORMS_KEY: str = TRAIN_TRANSFORMS_KEY
        self.BENTOML_ECR_URI: str = BENTOML_ECR_URI  
        self.BENTOML_ECR_LINK: str = BENTOML_ECR_LINK
        self.BENTOML_ECR_REGION: str = BENTOML_ECR_REGION
        self.BENTOML_ECR_USERNAME: str = BENTOML_ECR_USERNAME