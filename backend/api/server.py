# File: omniverse_ai/api/server.py
"""
FastAPI server implementation for OmniSense AI
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import io
import time
from ..core import OmniSenseAPI

app = FastAPI(
    title="OmniSense AI API",
    description="Multimodal Contextual Intelligence Platform",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the API processor
api_processor = OmniSenseAPI()

class TextRequest(BaseModel):
    text: str
    question: Optional[str] = None

@app.post("/analyze")
async def analyze(
    text: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    document: Optional[UploadFile] = File(None),
    objects: Optional[str] = Form(None)
):
    """
    Main analysis endpoint for multimodal processing
    
    Parameters:
    - text: Input text
    - question: Question about the input
    - image: Image file
    - audio: Audio file
    - document: Document file (PDF or image)
    - objects: Comma-separated list of objects to detect
    
    Returns:
    - Processing results
    """
    try:
        start_time = time.time()
        
        # Prepare request data
        request_data = {}
        
        if text:
            request_data["text"] = text
        if question:
            request_data["question"] = question
        if objects:
            request_data["objects"] = [obj.strip() for obj in objects.split(",")]
        
        # Process file uploads
        if image:
            request_data["image"] = await image.read()
        if audio:
            request_data["audio"] = await audio.read()
        if document:
            request_data["document"] = await document.read()
        
        # Process the request
        result = api_processor.process_request(request_data)
        processing_time = time.time() - start_time
        
        if result["success"]:
            return JSONResponse({
                **result,
                "processing_time": processing_time
            })
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)