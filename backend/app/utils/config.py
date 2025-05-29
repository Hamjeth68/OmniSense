import os
from typing import Optional
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "OmniSense AI"
    
    # HuggingFace Configuration
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    
    # Model Configuration
    VQA_MODEL: str = "dandelin/vilt-b32-finetuned-vqa"
    DOCUMENT_QA_MODEL: str = "impira/layoutlm-document-qa"
    OBJECT_DETECTION_MODEL: str = "google/owlvit-base-patch32"
    TEXT_CLASSIFICATION_MODEL: str = "facebook/bart-large-mnli"
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./omnisense.db")
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/jpg"]
    ALLOWED_AUDIO_TYPES: list = ["audio/wav", "audio/mp3", "audio/mpeg"]
    ALLOWED_DOCUMENT_TYPES: list = ["application/pdf", "image/jpeg", "image/png"]
    
    # Model Cache Configuration
    MODEL_CACHE_DIR: str = "./model_cache"
    
    # Processing Configuration
    MAX_IMAGE_SIZE: int = 1024
    AUDIO_SAMPLE_RATE: int = 16000
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()