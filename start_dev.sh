#!/bin/bash

echo "🚀 OmniSense AI - Quick Setup Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python $PYTHON_VERSION found"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version | cut -d' ' -f2)
    print_status "Python $PYTHON_VERSION found"
    PYTHON_CMD="python"
else
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_status "Node.js $NODE_VERSION found"
else
    print_error "Node.js not found. Please install Node.js 16+"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_status "npm $NPM_VERSION found"
else
    print_error "npm not found. Please install npm"
    exit 1
fi

echo ""
echo "Setting up OmniSense AI..."

# Create project structure
print_info "Creating project structure..."
mkdir -p omnisense-ai/{backend/app/{models,routers,utils},frontend}
cd omnisense-ai

# Initialize git
git init > /dev/null 2>&1
print_status "Git repository initialized"

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.env
instance/
.pytest_cache/

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.env.local
.env.development.local
.env.test.local
.env.production.local
build/
dist/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Models cache
model_cache/
.cache/
uploads/
EOF

print_status ".gitignore created"

# Setup backend
echo ""
print_info "Setting up backend..."
cd backend

# Create Python virtual environment
$PYTHON_CMD -m venv venv
print_status "Python virtual environment created"

# Activate virtual environment
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
transformers==4.35.2
torch==2.1.0
torchvision==0.16.0
Pillow==10.1.0
numpy==1.24.3
requests==2.31.0
python-dotenv==1.0.0
pydantic==2.5.0
huggingface-hub==0.19.4
datasets==2.14.6
accelerate==0.24.1
scipy==1.11.4
aiofiles==23.2.1
EOF

print_status "requirements.txt created"

# Create __init__.py files
touch app/__init__.py
touch app/models/__init__.py
touch app/routers/__init__.py
touch app/utils/__init__.py

print_status "Backend structure created"

# Create .env file
cat > .env << 'EOF'
# HuggingFace Configuration (Optional)
HUGGINGFACE_TOKEN=

# API Configuration
API_V1_STR=/api/v1
PROJECT_NAME=OmniSense AI

# File Upload Limits
MAX_FILE_SIZE=10485760

# Development
DEBUG=True
EOF

print_status "Backend .env created"

# Install Python dependencies
print_info "Installing Python dependencies (this may take a few minutes)..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1

if [ $? -eq 0 ]; then
    print_status "Python dependencies installed"
else
    print_warning "Some Python dependencies may have failed to install"
fi

# Setup frontend
echo ""
print_info "Setting up frontend..."
cd ../frontend

# Check if React app already exists
if [ ! -f "package.json" ]; then
    print_info "Creating React application..."
    npx create-react-app . > /dev/null 2>&1
    print_status "React application created"
fi

# Install additional dependencies
print_info "Installing additional frontend dependencies..."
npm install axios react-dropzone lucide-react > /dev/null 2>&1
npm install -D tailwindcss postcss autoprefixer > /dev/null 2>&1

if [ $? -eq 0 ]; then
    print_status "Frontend dependencies installed"
else
    print_warning "Some frontend dependencies may have failed to install"
fi

# Initialize Tailwind
npx tailwindcss init -p > /dev/null 2>&1
print_status "Tailwind CSS initialized"

# Create frontend .env
cat > .env << 'EOF'
REACT_APP_API_URL=http://localhost:8000/api/v1
EOF

print_status "Frontend .env created"

# Create startup script
cd ..
cat > start_dev.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting OmniSense AI Development Environment..."

# Check if backend virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "❌ Backend virtual environment not found. Please run setup first."
    exit 1
fi

# Start backend
echo "📡 Starting backend server..."
cd backend
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend server..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Development servers started successfully!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ All servers stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
EOF

chmod +x start_dev.sh
print_status "Development startup script created"

# Create installation completion message
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Get a free HuggingFace token (optional but recommended):"
echo "   https://huggingface.co/settings/tokens"
echo "   Add it to backend/.env file"
echo ""
echo "2. Start development servers:"
echo "   ./start_dev.sh"
echo ""
echo "3. Or start manually:"
echo "   Backend: cd backend && source venv/bin/activate && python -m uvicorn app.main:app --reload"
echo "   Frontend: cd frontend && npm start"
echo ""
print_warning "Note: First run will download AI models (~2-4GB). This is normal and happens once."
echo ""
print_info "Visit http://localhost:3000 when servers are running"