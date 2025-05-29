import React from 'react';
import { CheckCircle, XCircle, Eye, FileText, Search } from 'lucide-react';

const ResultDisplay = ({ result, loading }) => {
  if (loading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-2/3"></div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <div className="flex items-center mb-4">
        {result.success ? (
          <CheckCircle className="h-6 w-6 text-green-500" />
        ) : (
          <XCircle className="h-6 w-6 text-red-500" />
        )}
        <h3 className="ml-2 text-lg font-semibold">
          {result.success ? 'Analysis Complete' : 'Analysis Failed'}
        </h3>
      </div>

      {result.success && result.data && (
        <>
          {/* Summary */}
          <div className="mb-6">
            <h4 className="font-medium text-gray-900 mb-2">Summary</h4>
            <p className="text-gray-700 bg-gray-50 p-3 rounded">
              {result.data.summary}
            </p>
          </div>

          {/* Confidence Score */}
          <div className="mb-6">
            <h4 className="font-medium text-gray-900 mb-2">Confidence Score</h4>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full"
                style={{ width: `${(result.data.confidence * 100).toFixed(1)}%` }}
              ></div>
            </div>
            <span className="text-sm text-gray-600">
              {(result.data.confidence * 100).toFixed(1)}%
            </span>
          </div>

          {/* Detailed Results */}
          <div className="space-y-4">
            {result.data.results.vqa && (
              <ResultCard
                icon={<Eye className="h-5 w-5" />}
                title="Visual Question Answering"
                data={result.data.results.vqa}
              />
            )}
            
            {result.data.results.object_detection && (
              <ResultCard
                icon={<Search className="h-5 w-5" />}
                title="Object Detection"
                data={result.data.results.object_detection}
              />
            )}
            
            {result.data.results.document_qa && (
              <ResultCard
                icon={<FileText className="h-5 w-5" />}
                title="Document Analysis"
                data={result.data.results.document_qa}
              />
            )}
            
            {result.data.results.image_description && (
              <ResultCard
                icon={<Eye className="h-5 w-5" />}
                title="Image Description"
                data={result.data.results.image_description}
              />
            )}
          </div>
        </>
      )}

      {!result.success && (
        <div className="text-red-600">
          <p>{result.message}</p>
        </div>
      )}
    </div>
  );
};

const ResultCard = ({ icon, title, data }) => (
  <div className="border rounded-lg p-4">
    <div className="flex items-center mb-2">
      {icon}
      <h5 className="ml-2 font-medium">{title}</h5>
    </div>
    
    {data.error ? (
      <p className="text-red-600 text-sm">{data.error}</p>
    ) : (
      <div className="text-sm text-gray-700">
        {data.answer && (
          <p><span className="font-medium">Answer:</span> {data.answer}</p>
        )}
        {data.confidence && (
          <p><span className="font-medium">Confidence:</span> {(data.confidence * 100).toFixed(1)}%</p>
        )}
        {data.detections && (
          <div>
            <p className="font-medium mb-1">Detected Objects ({data.count}):</p>
            <ul className="list-disc list-inside space-y-1">
              {data.detections.slice(0, 5).map((detection, idx) => (
                <li key={idx}>
                  {detection.label} ({(detection.confidence * 100).toFixed(1)}%)
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    )}
  </div>
);

export default ResultDisplay;