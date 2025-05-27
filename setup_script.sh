#!/bin/bash

# RAG Application Setup Script
echo "🚀 Setting up RAG Application with Ollama, Chroma, and Streamlit"
echo "================================================================"

# Create project directory structure
echo "📁 Creating project directory structure..."
mkdir -p rag-app/streamlit_app
cd rag-app

# Create docker-compose.yml
echo "📝 Creating docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: rag_ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - rag_network

  chroma:
    image: chromadb/chroma:latest
    container_name: rag_chroma
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - rag_network

  streamlit:
    build:
      context: ./streamlit_app
      dockerfile: Dockerfile
    container_name: rag_streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./streamlit_app:/app
      - uploaded_docs:/app/uploaded_docs
    environment:
      - OLLAMA_URL=http://ollama:11434
      - CHROMA_URL=http://chroma:8000
    depends_on:
      ollama:
        condition: service_healthy
      chroma:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - rag_network

volumes:
  ollama_data:
  chroma_data:
  uploaded_docs:

networks:
  rag_network:
    driver: bridge
EOF

# Create streamlit app directory and files
echo "📝 Creating Streamlit application files..."

# Create Dockerfile for Streamlit
cat > streamlit_app/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for uploaded documents
RUN mkdir -p uploaded_docs

# Expose port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
EOF

# Create requirements.txt
cat > streamlit_app/requirements.txt << 'EOF'
streamlit==1.29.0
chromadb==0.4.18
ollama==0.1.7
langchain==0.0.350
langchain-community==0.0.1
pypdf2==3.0.1
python-docx==1.1.0
python-multipart==0.0.6
sentence-transformers==2.2.2
numpy==1.24.3
pandas==2.1.4
tiktoken==0.5.2
EOF

echo "✅ Project structure created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Copy the Python files (app.py, rag_service.py, document_processor.py) to the streamlit_app/ directory"
echo "2. Run: docker compose up -d"
echo "3. Wait for services to be ready (check with: docker-compose logs -f)"
echo "4. Access the application at: http://localhost:8501"
echo ""
echo "🔧 Useful commands:"
echo "- View logs: docker compose logs -f"
echo "- Stop services: docker compose down"
echo "- Rebuild: docker compose up --build"
echo "- Check status: docker compose ps"
echo ""
echo "📚 Default models to try:"
echo "- llama2 (7B) - Good general purpose model"
echo "- mistral (7B) - Fast and efficient"
echo "- codellama (7B) - Good for code-related queries"
echo ""
echo "⚠️  Note: First time pulling models will take several minutes depending on your internet connection."
EOF
