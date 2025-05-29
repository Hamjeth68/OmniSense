import cv2
import numpy as np
import librosa
from PIL import Image
from transformers import pipeline
from typing import Dict, Any, Optional
import io
import base64

class InputProcessor:
    def __init__(self):
        self.modality_detector = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli"
        )
        
    def detect_modality(self, input_data: bytes, filename: str = "") -> str:
        """Detect input modality based on file content and name"""
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ""
        
        # Simple file extension based detection
        if file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
            return "image"
        elif file_extension in ['wav', 'mp3', 'mpeg', 'aac', 'flac']:
            return "audio"
        elif file_extension in ['pdf', 'doc', 'docx', 'txt']:
            return "document"
        elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
            return "video"
        else:
            return "text"
    
    def preprocess_image(self, image_bytes: bytes) -> Image.Image:
        """Preprocess image data"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (max 1024x1024)
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def preprocess_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Preprocess audio data"""
        try:
            # Load audio from bytes
            audio_data, sample_rate = librosa.load(
                io.BytesIO(audio_bytes), sr=16000
            )
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            return audio_data
        except Exception as e:
            raise ValueError(f"Error processing audio: {str(e)}")
    
    def preprocess_document(self, document_bytes: bytes) -> Image.Image:
        """Preprocess document data (convert to image)"""
        try:
            # For PDF, convert first page to image
            if document_bytes.startswith(b'%PDF'):
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(document_bytes, first_page=1, last_page=1)
                return images[0] if images else None
            else:
                # Treat as image
                return self.preprocess_image(document_bytes)
        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}")
    
    def preprocess(self, input_data: bytes, modality: str, filename: str = "") -> Any:
        """Main preprocessing function"""
        if modality == "image":
            return self.preprocess_image(input_data)
        elif modality == "audio":
            return self.preprocess_audio(input_data)
        elif modality == "document":
            return self.preprocess_document(input_data)
        else:
            return input_data.decode('utf-8') if isinstance(input_data, bytes) else input_data