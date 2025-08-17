"""
Configuration settings for the AI Chatbot application
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model paths - keeping your existing paths
    MISTRAL_MODEL_PATH: str = "../../../local_models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    SD_MODEL_PATH: str = "../../../local_models/stable-diffusion/v1-5-pruned-emaonly.safetensors"
    
    # Model parameters optimized for RTX 3050
    MISTRAL_N_GPU_LAYERS: int = 20
    MISTRAL_N_CTX: int = 4096
    MISTRAL_N_THREADS: int = 8
    
    # Image generation parameters
    SD_NUM_INFERENCE_STEPS: int = 20
    SD_GUIDANCE_SCALE: float = 7.5
    SD_WIDTH: int = 512
    SD_HEIGHT: int = 512
    
    # Memory settings
    MAX_CONVERSATION_TOKENS: int = 1000
    
    # Supported languages
    SUPPORTED_LANGUAGES: list = ["en", "hi", "es", "fr"]
    
    # Paths
    DATA_DIR: str = "data"
    CONVERSATIONS_DIR: str = "data/conversations"
    IMAGES_DIR: str = "data/generated_images"
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        Path(self.DATA_DIR).mkdir(exist_ok=True)
        Path(self.CONVERSATIONS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.IMAGES_DIR).mkdir(parents=True, exist_ok=True)
