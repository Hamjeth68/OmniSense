from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

from ..models.input_processor import InputProcessor
from ..models.multimodal_fusion import MultimodalFusion
from ..models.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])

# Initialize models
processor = InputProcessor()
fusion = MultimodalFusion()
response_generator = ResponseGenerator()

class AnalysisQuery(BaseModel):
    text: Optional[str] = None
    question: Optional[str] = None
    detection_labels: Optional[List[str]] = None

@router.post("/")
async def analyze_multimodal(
    query: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
    detection_labels: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    document: Optional[UploadFile] = File(None)
):
    """
    Analyze multimodal inputs and return comprehensive results
    """
    try:
        inputs = {}
        
        # Process image
        if image:
            image_bytes = await image.read()
            modality = processor.detect_modality(image_bytes, image.filename)
            if modality == "image":
                inputs["image"] = processor.preprocess_image(image_bytes)
            else:
                raise HTTPException(400, "Invalid image file")
        
        # Process document
        if document:
            doc_bytes = await document.read()
            inputs["document"] = processor.preprocess_document(doc_bytes)
        
        # Process audio (placeholder for future implementation)
        if audio:
            audio_bytes = await audio.read()
            inputs["audio"] = processor.preprocess_audio(audio_bytes)
        
        # Add text inputs
        if question:
            inputs["question"] = question
        if query:
            inputs["query"] = query
        if detection_labels:
            try:
                inputs["detection_labels"] = detection_labels.split(",")
            except:
                inputs["detection_labels"] = [detection_labels]
        
        # Process through fusion engine
        results = fusion.process(inputs)
        
        # Generate response
        response = response_generator.generate_response(results, query or question or "")
        
        return response_generator.format_for_api(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return response_generator.format_error(str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "OmniSense AI is running"}

@router.get("/models/status")
async def models_status():
    """Check status of loaded models"""
    return {
        "vqa_model": fusion.vqa_model is not None,
        "object_detector": fusion.object_detector is not None,
        "document_qa_model": fusion.document_qa_model is not None
    }