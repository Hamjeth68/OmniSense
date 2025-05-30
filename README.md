# OmniSense AI 

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Local Setup](#local-setup)
5. [Backend Implementation](#backend-implementation)
6. [Frontend Implementation](#frontend-implementation)
7. [Running Locally](#running-locally)
8. [Deployment Options](#deployment-options)
9. [Troubleshooting](#troubleshooting)

## System Architecture


## System Components

### 1. Client Layer
- **Frontend Interfaces** (Flutter):
  - Cross-platform Desktop/Mobile/Web apps
  - Unified UI for all user interactions
  - Real-time updates via WebSockets

### 2. API Gateway
- **FastAPI**:
  - RESTful endpoint management
  - JWT Authentication & rate-limiting
  - Request routing to microservices
  - OpenAPI/Swagger documentation

### 3. Task Processing
- **Celery Workers**:
  - Distributed task queue (video/image/text/audio)
  - Priority-based workload distribution
- **Redis**:
  - Cache layer for frequent queries
  - Celery message broker
  - Rate-limiting storage

![Architecture Diagram](system_architecture_diagram.html)

## Prerequisites

### System Requirements
- Python 3.8+
- Node.js 16+
- Docker (optional)
- 8GB+ RAM (for running ML models)
- CUDA-compatible GPU (recommended)

### Required Accounts
- GitHub (for code hosting)
- HuggingFace (for model access)
- Vercel/Netlify (frontend deployment)
- Railway/Render (backend deployment)

## Project Structure

```
omnisense-ai/
├── backend/               # FastAPI backend
│   ├── app/               # Application code
│   │   ├── models/        # ML models
│   │   ├── routers/       # API endpoints
│   │   └── utils/         # Configuration
│   ├── Dockerfile         # Container config
│   └── requirements.txt   # Python dependencies
├── frontend/              # React frontend
│   ├── src/               # React components
│   ├── public/            # Static assets
│   └── package.json       # JS dependencies
└── docker-compose.yml     # Multi-container setup
```

## Local Setup

### 1. Clone and Initialize Project
```bash
git clone https://github.com/your-repo/omnisense-ai.git
cd omnisense-ai
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

## Backend Implementation

Key components:
- **FastAPI** for REST endpoints
- **HuggingFace Transformers** for ML models
- **Redis** for caching
- **Celery** for async tasks

### Configuration
Set up `.env` with:
```env
HUGGINGFACE_TOKEN=your_token
DATABASE_URL=postgresql://user:pass@localhost/omnisense
REDIS_URL=redis://localhost:6379
```

## Frontend Implementation

Built with:
- **React** for UI components
- **Tailwind CSS** for styling
- **Lucide** for icons
- **Axios** for API calls

Main features:
- File upload for images/documents
- Question input for VQA
- Object detection configuration
- Results visualization

## Running Locally

### Start Backend
```bash
cd backend
uvicorn app.main:app --reload
```

### Start Frontend
```bash
cd frontend
npm start
```

Access at:
- Frontend: `http://localhost:3000`
- API Docs: `http://localhost:8000/docs`

## Deployment Options

### Backend
1. **Railway** (recommended):
   ```bash
   railway login
   railway up
   ```

2. **Render**:
   - Connect GitHub repo
   - Set build/start commands

### Frontend
1. **Vercel**:
   ```bash
   npm install -g vercel
   vercel
   ```

2. **Netlify**:
   - Connect repo
   - Set build command: `npm run build`

## Troubleshooting

Common issues:
- **Model loading errors**: Clear HuggingFace cache
- **CORS issues**: Verify API_URL and CORS settings
- **Memory issues**: Use smaller models or enable GPU

```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/
```

For full implementation details and code samples, refer to the complete guide in the repository. The modular architecture allows easy extension with additional models and features.
