from transformers import pipeline
from PIL import Image
import torch
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class MultimodalFusion:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self._load_models()
    
    def _load_models(self):
        """Load all required models"""
        try:
            self.vqa_model = pipeline(
                "visual-question-answering",
                model="dandelin/vilt-b32-finetuned-vqa",
                device=self.device
            )
            logger.info("VQA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading VQA model: {e}")
            self.vqa_model = None
        
        try:
            self.object_detector = pipeline(
                "zero-shot-object-detection",
                model="google/owlvit-base-patch32",
                device=self.device
            )
            logger.info("Object detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading object detection model: {e}")
            self.object_detector = None
        
        try:
            self.document_qa_model = pipeline(
                "document-question-answering",
                model="impira/layoutlm-document-qa",
                device=self.device
            )
            logger.info("Document QA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading document QA model: {e}")
            self.document_qa_model = None
    
    def visual_question_answering(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Perform visual question answering"""
        if not self.vqa_model:
            return {"error": "VQA model not available"}
        
        try:
            result = self.vqa_model(image=image, question=question)
            return {
                "answer": result[0]["answer"],
                "confidence": result[0]["score"],
                "type": "visual_qa"
            }
        except Exception as e:
            logger.error(f"Error in VQA: {e}")
            return {"error": f"VQA processing failed: {str(e)}"}
    
    def object_detection(self, image: Image.Image, candidate_labels: List[str] = None) -> Dict[str, Any]:
        """Perform zero-shot object detection"""
        if not self.object_detector:
            return {"error": "Object detection model not available"}
        
        if not candidate_labels:
            candidate_labels = ["person", "car", "chair", "table", "dog", "cat", "book", "phone", "computer"]
        
        try:
            results = self.object_detector(image, candidate_labels=candidate_labels)
            
            # Format results
            detections = []
            for result in results:
                detections.append({
                    "label": result["label"],
                    "confidence": result["score"],
                    "box": result["box"]
                })
            
            return {
                "detections": detections,
                "count": len(detections),
                "type": "object_detection"
            }
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return {"error": f"Object detection failed: {str(e)}"}
    
    def document_question_answering(self, document_image: Image.Image, question: str) -> Dict[str, Any]:
        """Perform document question answering"""
        if not self.document_qa_model:
            return {"error": "Document QA model not available"}
        
        try:
            result = self.document_qa_model(image=document_image, question=question)
            return {
                "answer": result["answer"],
                "confidence": result["score"],
                "type": "document_qa"
            }
        except Exception as e:
            logger.error(f"Error in document QA: {e}")
            return {"error": f"Document QA failed: {str(e)}"}
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing function that handles multiple modalities"""
        results = {}
        
        # Visual Question Answering
        if "image" in inputs and "question" in inputs:
            results["vqa"] = self.visual_question_answering(
                inputs["image"], 
                inputs["question"]
            )
        
        # Object Detection
        if "image" in inputs:
            candidate_labels = inputs.get("detection_labels", None)
            results["object_detection"] = self.object_detection(
                inputs["image"], 
                candidate_labels
            )
        
        # Document Question Answering
        if "document" in inputs and "question" in inputs:
            results["document_qa"] = self.document_question_answering(
                inputs["document"], 
                inputs["question"]
            )
        
        # Image Description (using VQA with generic question)
        if "image" in inputs and "question" not in inputs:
            results["image_description"] = self.visual_question_answering(
                inputs["image"], 
                "What is in this image?"
            )
        
        return results
    