from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

# Import models (we'll create these)
from ..models.input_processor import InputProcessor
from ..models.multimodal_fusion import MultimodalFusion
from ..models.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])

# Initialize models (lazy loading to avoid startup delays)
processor = None
fusion = None
response_generator = None

def get_models():
    """Lazy load models to avoid startup delays"""
    global processor, fusion, response_generator
    
    if processor is None:
        logger.info("Initializing models...")
        processor = InputProcessor()
        fusion = MultimodalFusion()
        response_generator = ResponseGenerator()
        logger.info("Models initialized successfully")
    
    return processor, fusion, response_generator

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
        # Get initialized models
        proc, fusion_engine, resp_gen = get_models()
        
        inputs = {}
        
        # Validate at least one input is provided
        if not any([image, document, audio, question, query]):
            raise HTTPException(
                status_code=400, 
                detail="Please provide at least one input (image, document, audio, or question)"
            )
        
        # Process image
        if image:
            if image.size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
            
            image_bytes = await image.read()
            modality = proc.detect_modality(image_bytes, image.filename)
            if modality == "image":
                inputs["image"] = proc.preprocess_image(image_bytes)
            else:
                raise HTTPException(status_code=400, detail="Invalid image file format")
        
        # Process document
        if document:
            if document.size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="Document file too large (max 10MB)")
            
            doc_bytes = await document.read()
            inputs["document"] = proc.preprocess_document(doc_bytes)
        
        # Process audio (placeholder for future implementation)
        if audio:
            if audio.size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="Audio file too large (max 10MB)")
            
            audio_bytes = await audio.read()
            inputs["audio"] = proc.preprocess_audio(audio_bytes)
        
        # Add text inputs
        if question:
            inputs["question"] = question.strip()
        if query:
            inputs["query"] = query.strip()
        if detection_labels:
            try:
                inputs["detection_labels"] = [label.strip() for label in detection_labels.split(",")]
            except:
                inputs["detection_labels"] = [detection_labels.strip()]
        
        # Process through fusion engine
        results = fusion_engine.process(inputs)
        
        # Generate response
        response = resp_gen.generate_response(results, query or question or "")
        
        return resp_gen.format_for_api(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        error_response = ResponseGenerator().format_error(f"Analysis failed: {str(e)}")
        return error_response

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Analysis service is running"}

@router.get("/models/status")
async def models_status():
    """Check status of loaded models"""
    try:
        # Try to get models (will initialize if not already done)
        proc, fusion_engine, resp_gen = get_models()
        
        return {
            "status": "loaded",
            "models": {
                "vqa_model": fusion_engine.vqa_model is not None,
                "object_detector": fusion_engine.object_detector is not None,
                "document_qa_model": fusion_engine.document_qa_model is not None
            },
            "processor": proc is not None,
            "response_generator": resp_gen is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "models": {
                "vqa_model": False,
                "object_detector": False,
                "document_qa_model": False
            }
        }

@router.get("/supported-formats")
async def supported_formats():
    """Get supported file formats and limits"""
    return {
        "image": {
            "formats": ["jpeg", "jpg", "png", "gif"],
            "max_size": "10MB"
        },
        "document": {
            "formats": ["pdf", "jpeg", "jpg", "png"],
            "max_size": "10MB"
        },
        "audio": {
            "formats": ["wav", "mp3", "m4a"],
            "max_size": "10MB",
            "note": "Audio processing coming soon"
        }
    }