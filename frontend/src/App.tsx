import React, { useState } from 'react';
import axios from 'axios';
import FileUpload from './components/FileUpload';
import ResultDisplay from './components/ResultDisplay';
import { Brain, Upload, MessageSquare } from 'lucide-react';

interface AnalysisResult {
  success: boolean;
  message?: string;
  data?: {
    summary: string;
    confidence: number;
    results: {
      vqa?: {
        answer: string;
        confidence: number;
      };
      object_detection?: {
        detections: Array<{
          label: string;
          confidence: number;
        }>;
        count: number;
      };
      document_qa?: {
        answer: string;
        confidence: number;
      };
      image_description?: {
        description: string;
      };
    };
  };
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<File | null>(null);
  const [selectedAudio, setSelectedAudio] = useState<File | null>(null);
  const [question, setQuestion] = useState('');
  const [detectionLabels, setDetectionLabels] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!selectedImage && !selectedDocument && !selectedAudio && !question.trim()) {
      alert('Please provide at least one input (image, document, audio, or question)');
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    
    if (selectedImage) formData.append('image', selectedImage);
    if (selectedDocument) formData.append('document', selectedDocument);
    if (selectedAudio) formData.append('audio', selectedAudio);
    if (question.trim()) formData.append('question', question.trim());
    if (detectionLabels.trim()) formData.append('detection_labels', detectionLabels.trim());

    try {
      const response = await axios.post<AnalysisResult>(`${API_BASE_URL}/analyze/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000,
      });
      setResult(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      setResult({
        success: false,
        message: axios.isAxiosError(error) 
          ? error.response?.data?.message || 'Analysis failed. Please try again.'
          : 'Analysis failed. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setSelectedDocument(null);
    setSelectedAudio(null);
    setQuestion('');
    setDetectionLabels('');
    setResult(null);
  };

  const imageTypes = {
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png'],
    'image/gif': ['.gif']
  };

  const documentTypes = {
    'application/pdf': ['.pdf'],
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png']
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header and Main Content remain the same as original JSX */}
      {/* ... */}
    </div>
  );
}

export default App;