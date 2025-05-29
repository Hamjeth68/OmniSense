
import os
import logging
import time
from typing import Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import io
import cv2
import torch
from transformers import (
    OwlViTProcessor,
    ViltForQuestionAnswering,
    ViltProcessor,
    LayoutLMForQuestionAnswering,
    LayoutLMTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from transformers.pipelines import pipeline
from transformers.models.owlvit import OwlViTForObjectDetection
import soundfile as sf
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalityDetector:
    """Enhanced modality detection with confidence scoring"""
    def __init__(self):
        self.model = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        self.modality_labels = [
            "text", "image", "audio", 
            "video", "document", "3d model"
        ]
    
    def detect(self, input_data: Union[str, bytes]) -> Dict[str, Any]:
        """
        Detect the modality of input data with confidence scores
        
        Args:
            input_data: Input data to analyze (text string or binary data)
            
        Returns:
            Dictionary with modality detection results
        """
        try:
            if isinstance(input_data, bytes):
                # Try to determine binary data type
                if self._is_image(input_data):
                    return {"modality": "image", "confidence": 1.0}
                elif self._is_audio(input_data):
                    return {"modality": "audio", "confidence": 1.0}
                elif self._is_pdf(input_data):
                    return {"modality": "document", "confidence": 1.0}
                # Fall back to text analysis for first 1KB if possible
                try:
                    sample = input_data[:1024].decode('utf-8', errors='ignore')
                    if len(sample.strip()) > 20:  # Minimum viable text length
                        result = self.model(sample, self.modality_labels)
                        return {
                            "modality": result["labels"][0],
                            "confidence": result["scores"][0],
                            "alternatives": list(zip(result["labels"], result["scores"]))
                        }
                except UnicodeDecodeError:
                    pass
                return {"modality": "unknown", "confidence": 0.0}
            else:
                # Text-based detection
                sample = str(input_data)[:1024]
                result = self.model(sample, self.modality_labels)
                return {
                    "modality": result["labels"][0],
                    "confidence": result["scores"][0],
                    "alternatives": list(zip(result["labels"], result["scores"]))
                }
        except Exception as e:
            logger.error(f"Modality detection failed: {str(e)}")
            return {"modality": "error", "confidence": 0.0, "error": str(e)}
    
    def _is_image(self, data: bytes) -> bool:
        """Check if data is an image"""
        try:
            Image.open(io.BytesIO(data))
            return True
        except:
            return False
    
    def _is_audio(self, data: bytes) -> bool:
        """Check if data is audio"""
        try:
            with sf.SoundFile(io.BytesIO(data)) as f:
                return True
        except:
            try:
                librosa.load(io.BytesIO(data), sr=None)
                return True
            except:
                return False
    
    def _is_pdf(self, data: bytes) -> bool:
        """Check if data is PDF"""
        return data[:4] == b'%PDF'

class InputPreprocessor:
    """Advanced input preprocessing for all modalities"""
    def __init__(self):
        self.image_size = (512, 512)
        self.audio_sample_rate = 16000
        self.text_max_length = 512
    
    def preprocess(self, data: Any, modality: str) -> Any:
        """
        Preprocess input data based on modality
        
        Args:
            data: Input data
            modality: One of 'text', 'image', 'audio', 'video', 'document'
            
        Returns:
            Preprocessed data ready for model input
        """
        processor = getattr(self, f"_process_{modality}", None)
        if processor:
            return processor(data)
        raise ValueError(f"Unsupported modality: {modality}")
    
    def _process_text(self, text: str) -> Dict[str, Any]:
        """Preprocess text input"""
        return {
            "text": str(text)[:self.text_max_length],
            "length": len(str(text))
        }
    def _process_image(self, image: Union[str, bytes, bytearray, memoryview, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Preprocess image input"""
        # Convert bytearray or memoryview to bytes
        if isinstance(image, (bytearray, memoryview)):
            image = bytes(image)
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            if os.path.isfile(image):
                image = Image.open(image)
            else:  # Assume it's a base64 string
                import base64
                image = Image.open(io.BytesIO(base64.b64decode(image)))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pass  # Already an Image.Image object
        else:
            raise ValueError("Unsupported image input type: {}".format(type(image)))
        
        # Convert to RGB if needed
        if hasattr(image, "mode") and image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and normalize
        image = np.array(image.resize(self.image_size))
        image = image.astype(np.float32) / 255.0
        
        return {
            "pixels": image,
            "shape": image.shape,
            "format": "RGB"
        }
    
    def _process_audio(self, audio: Union[str, bytes]) -> Dict[str, Any]:
        """Preprocess audio input"""
        waveform = None
        sr = self.audio_sample_rate
        if isinstance(audio, bytes):
            waveform, sr = librosa.load(io.BytesIO(audio), sr=self.audio_sample_rate)
        elif isinstance(audio, str):
            if os.path.isfile(audio):
                waveform, sr = librosa.load(audio, sr=self.audio_sample_rate)
            else:  # Assume it's a base64 string
                import base64
                decoded_audio = base64.b64decode(audio)
                waveform, sr = librosa.load(io.BytesIO(decoded_audio), sr=self.audio_sample_rate)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
        mel = librosa.feature.melspectrogram(y=waveform, sr=sr)
        
        return {
            "waveform": waveform,
            "sample_rate": sr,
            "mfcc": mfcc,
            "chroma": chroma,
            "mel": mel
        }
    
    def _process_document(self, document: Union[str, bytes]) -> Dict[str, Any]:
        """Preprocess document input"""
        if isinstance(document, bytes):
            # For PDFs, extract text and layout
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(document)) as pdf:
                    text = "\n".join(page.extract_text() for page in pdf.pages)
                    return {
                        "text": text,
                        "pages": len(pdf.pages),
                        "format": "pdf"
                    }
            except:
                # Fall back to OCR if needed
                try:
                    import pytesseract
                    image = Image.open(io.BytesIO(document))
                    text = pytesseract.image_to_string(image)
                    return {
                        "text": text,
                        "pages": 1,
                        "format": "image"
                    }
                except:
                    return {"text": "", "error": "Could not extract text"}
        else:
            # Assume it's text or file path
            if os.path.isfile(document):
                with open(document, 'r') as f:
                    return {"text": f.read(), "format": "text"}
            else:
                return {"text": str(document), "format": "text"}

class MultimodalFusionEngine:
    """Core fusion engine for combining multiple modalities"""
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self._load_models()
        self.cache = {}  # Simple in-memory cache for demo
        
    def _load_models(self) -> Dict[str, Any]:
        """Load all required models"""
        models = {}
        
        # Visual Question Answering
        models["vqa"] = {
            "model": ViltForQuestionAnswering.from_pretrained(
                "dandelin/vilt-b32-finetuned-vqa"
            ).to(self.device),
            "processor": ViltProcessor.from_pretrained(
                "dandelin/vilt-b32-finetuned-vqa"
            )
        }
        
        # Document Question Answering
        models["doc_qa"] = {
            "model": LayoutLMForQuestionAnswering.from_pretrained(
                "impira/layoutlm-document-qa"
            ).to(self.device),
            "tokenizer": LayoutLMTokenizer.from_pretrained(
                "impira/layoutlm-document-qa"
            )
        }
        
        # Zero-Shot Object Detection
        models["object_detection"] = {
            "model": OwlViTForObjectDetection.from_pretrained(
                "google/owlvit-base-patch32"
            ).to(self.device),
            "processor": OwlViTProcessor.from_pretrained(
                "google/owlvit-base-patch32"
            )
        }
        
        # Text Classification (for modality understanding)
        models["text_classification"] = {
            "model": AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased"
            ).to(self.device),
            "tokenizer": AutoTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
        }
        
        return models
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multimodal inputs and return fused results
        
        Args:
            inputs: Dictionary containing inputs of different modalities
                   (e.g., {"image": image_data, "text": "What is this?"})
                   
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        try:
            # Visual Question Answering
            if "image" in inputs and "text" in inputs:
                cache_key = f"vqa_{hash(str(inputs['image']))}_{hash(inputs['text'])}"
                if cache_key in self.cache:
                    results["vqa"] = self.cache[cache_key]
                else:
                    vqa_inputs = self.models["vqa"]["processor"](
                        inputs["image"], 
                        inputs["text"], 
                        return_tensors="pt"
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.models["vqa"]["model"](**vqa_inputs)
                    answer = self.models["vqa"]["processor"].decode(
                        outputs.logits.argmax(-1).item()
                    )
                    results["vqa"] = {"answer": answer, "confidence": torch.softmax(outputs.logits, dim=-1).max().item()}
                    self.cache[cache_key] = results["vqa"]
            
            # Document Question Answering
            if "document" in inputs and "text" in inputs:
                cache_key = f"docqa_{hash(str(inputs['document']))}_{hash(inputs['text'])}"
                if cache_key in self.cache:
                    results["doc_qa"] = self.cache[cache_key]
                else:
                    doc_inputs = self.models["doc_qa"]["tokenizer"](
                        inputs["text"],
                        inputs["document"],
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.models["doc_qa"]["model"](**doc_inputs)
                    answer_start = torch.argmax(outputs.start_logits)
                    answer_end = torch.argmax(outputs.end_logits) + 1
                    answer = self.models["doc_qa"]["tokenizer"].decode(
                        doc_inputs.input_ids[0][answer_start:answer_end]
                    )
                    results["doc_qa"] = {
                        "answer": answer,
                        "confidence": (
                            torch.softmax(outputs.start_logits, dim=-1).max().item() *
                            torch.softmax(outputs.end_logits, dim=-1).max().item()
                        )
                    }
                    self.cache[cache_key] = results["doc_qa"]
            
            # Object Detection
            if "image" in inputs and "objects" in inputs:
                cache_key = f"objdet_{hash(str(inputs['image']))}_{hash(str(inputs['objects']))}"
                if cache_key in self.cache:
                    results["object_detection"] = self.cache[cache_key]
                else:
                    obj_inputs = self.models["object_detection"]["processor"](
                        text=inputs["objects"],
                        images=inputs["image"],
                        return_tensors="pt"
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.models["object_detection"]["model"](**obj_inputs)
                    
                    # Process outputs
                    target_sizes = torch.tensor([inputs["image"].size[::-1]])
                    results["object_detection"] = self.models["object_detection"]["processor"].post_process_object_detection(
                        outputs, 
                        target_sizes=target_sizes,
                        threshold=0.1
                    )
                    self.cache[cache_key] = results["object_detection"]
            
            # Text Classification
            if "text" in inputs and not ("image" in inputs or "document" in inputs):
                cache_key = f"textcls_{hash(inputs['text'])}"
                if cache_key in self.cache:
                    results["text_classification"] = self.cache[cache_key]
                else:
                    text_inputs = self.models["text_classification"]["tokenizer"](
                        inputs["text"],
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.models["text_classification"]["model"](**text_inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    results["text_classification"] = {
                        "labels": ["positive", "negative", "neutral"],
                        "scores": probs.cpu().numpy().tolist()[0]
                    }
                    self.cache[cache_key] = results["text_classification"]
        
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            results["error"] = str(e)
        
        return results

class OmniSenseAPI:
    """Main API controller for OmniSense AI"""
    def __init__(self):
        self.modality_detector = ModalityDetector()
        self.preprocessor = InputPreprocessor()
        self.fusion_engine = MultimodalFusionEngine()
        self.context_memory = []  # Simple context memory for demo
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming request with multimodal data
        
        Args:
            request_data: Dictionary containing request data
                         (e.g., {"text": "question", "image": image_data})
                         
        Returns:
            Dictionary with processing results
        """
        try:
            # Step 1: Detect modalities
            modalities = {}
            for key, data in request_data.items():
                if key in ["text", "question"]:
                    modalities["text"] = data
                else:
                    detection = self.modality_detector.detect(data)
                    modalities[detection["modality"]] = data
            
            # Step 2: Preprocess inputs
            preprocessed = {}
            for modality, data in modalities.items():
                preprocessed[modality] = self.preprocessor.preprocess(data, modality)
            
            # Step 3: Add context from memory
            if self.context_memory:
                preprocessed["context"] = self.context_memory[-1]  # Use last context
            
            # Step 4: Process with fusion engine
            results = self.fusion_engine.process(preprocessed)
            
            # Step 5: Update context memory
            self._update_context_memory(preprocessed, results)
            
            return {
                "success": True,
                "results": results,
                "modalities": list(modalities.keys())
            }
        except Exception as e:
            logger.error(f"Request processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _update_context_memory(self, inputs: Dict[str, Any], results: Dict[str, Any]):
        """Update the context memory with new information"""
        context = {
            "timestamp": time.time(),
            "inputs": {k: v for k, v in inputs.items() if k != "image"},
            "results": results
        }
        
        # Keep memory size manageable
        if len(self.context_memory) >= 10:
            self.context_memory.pop(0)
        self.context_memory.append(context)