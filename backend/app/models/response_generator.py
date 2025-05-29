from typing import Dict, Any, List
import json

class ResponseGenerator:
    def __init__(self):
        pass
    
    def generate_response(self, results: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """Generate a comprehensive response from multimodal results"""
        
        response = {
            "query": query,
            "timestamp": self._get_timestamp(),
            "results": results,
            "summary": self._generate_summary(results),
            "confidence": self._calculate_overall_confidence(results)
        }
        
        return response
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the results"""
        summary_parts = []
        
        # VQA Summary
        if "vqa" in results and "answer" in results["vqa"]:
            summary_parts.append(f"Visual Analysis: {results['vqa']['answer']}")
        
        # Object Detection Summary
        if "object_detection" in results and "detections" in results["object_detection"]:
            detections = results["object_detection"]["detections"]
            if detections:
                detected_objects = [det["label"] for det in detections[:3]]  # Top 3
                summary_parts.append(f"Detected objects: {', '.join(detected_objects)}")
        
        # Document QA Summary
        if "document_qa" in results and "answer" in results["document_qa"]:
            summary_parts.append(f"Document insight: {results['document_qa']['answer']}")
        
        # Image Description Summary
        if "image_description" in results and "answer" in results["image_description"]:
            summary_parts.append(f"Image description: {results['image_description']['answer']}")
        
        return " | ".join(summary_parts) if summary_parts else "No significant findings detected."
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        confidences = []
        
        for result_type, result_data in results.items():
            if isinstance(result_data, dict):
                if "confidence" in result_data:
                    confidences.append(result_data["confidence"])
                elif "detections" in result_data:
                    # For object detection, use average of top detections
                    det_confidences = [det.get("confidence", 0) for det in result_data["detections"][:3]]
                    if det_confidences:
                        confidences.append(sum(det_confidences) / len(det_confidences))
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def format_for_api(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format response for API consumption"""
        return {
            "success": True,
            "data": response,
            "message": "Analysis completed successfully"
        }
    
    def format_error(self, error_message: str) -> Dict[str, Any]:
        """Format error response"""
        return {
            "success": False,
            "data": None,
            "message": error_message,
            "timestamp": self._get_timestamp()
        }