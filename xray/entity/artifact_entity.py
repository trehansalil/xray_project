from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader

@dataclass
class DataIngestionArtifacts:
    train_file_path: str
    test_file_path: str

# @dataclass    
# class DataValidationArtifacts:    
#     imbalance_data_valid: bool
#     raw_data_valid: bool
    
@dataclass    
class DataTransformationArtifacts:    
    transformed_train_object: DataLoader
    transformed_test_object: DataLoader

    train_transform_file_path: str
    test_transform_file_path: str  
    
@dataclass    
class ModelTrainerArtifacts:    
    trained_model_path: str
    
@dataclass
class ModelEvaluationArtifacts:
    model_accuracy: float

@dataclass
class ModelPusherArtifacts:
    bentoml_model_name: str
    bentoml_service_name: str 