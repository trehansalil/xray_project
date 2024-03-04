import os
from typing import List
from datetime import datetime
import torch

# Requirement file name
requirement_file_name = "requirements_dev.txt"

# common constants
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
S3_DATA_FOLDER:str = 'data'
BUCKET_NAME = 'lungxraydataset'



# Data ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
TRAIN_DATA_DIR = "train"
TEST_DATA_DIR = "test"


# # Data validation constants
# IMBALANCE_DATA_DIR = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_IMBALANCE_DATA_DIR)
# RAW_DATA_DIR = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_RAW_DATA_DIR)
# IMBALANCE_DATA_COLUMNS = ['id', 'label', 'tweet']
# RAW_DATA_COLUMNS = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language',	'neither', 'class', 'tweet']

# Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
TRAIN_TRANSFORMS_FILE: str = "train_transforms.pkl"
TEST_TRANSFORMS_FILE: str = "test_transforms.pkl"

CLASS_LABEL_1: str = "NORMAL"
CLASS_LABEL_2: str = "PNEUMONIA"

BRIGHTNESS: int = 0.10
CONTRAST: int = 0.1
SATURATION: int = 0.10
HUE: int = 0.1
RESIZE: int = 224
CENTERCROP: int = 224
RANDOMROTATION: int = 10

BATCH_SIZE: int = 2
SHUFFLE: bool = False
PIN_MEMORY: bool = True

NORMALIZE_LIST_1: List[int] = [0.485, 0.456, 0.406]
NORMALIZE_LIST_2: List[int] = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS_KEY: str = "xray_train_transforms"

# Model Trainer constants
MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
TRAINED_MODEL_DIR = 'xray_model'
TRAINED_MODEL_NAME = 'model.pt'
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STEP_SIZE: int = 6
GAMMA: int = 0.5
EPOCH: int = 1
LEARNING_RATE = 0.01
MOMENTUM_BETA = 0.8
RANDOM_STATE = 42
BENTOML_MODEL_NAME: str = "xray_model"
BENTOML_SERVICE_NAME: str = "xray_service"
BENTOML_ECR_URI: str = "xray_bento_image"
BENTOML_ECR_REGION: str = "us-east-1"
BENTOML_ECR_LINK: str = f"430247671429.dkr.ecr.{BENTOML_ECR_REGION}.amazonaws.com"
BENTOML_ECR_USERNAME: str = "AWS"
PREDICTION_LABEL: dict = {"0": CLASS_LABEL_1, 1: CLASS_LABEL_2}

# # Model Arhcitecture constants
# MAX_WORDS = 5000
# MAX_LEN = 300
# LOSS = 'binary_crossentropy'
# METRICS = ['accuracy']
# ACTIVATION = 'sigmoid'

# # Model Evaluation constants
# MODEL_EVALUATION_ARTIFACTS_DIR = "ModelEvaluationArtifacts"
# BEST_MODEL_DIR = "best_Model"
# MODEL_EVALUATION_FILE_NAME = 'loss.csv'


# MODEL_NAME = 'model.h5'
# APP_HOST = '0.0.0.0'
# APP_PORT = 8080