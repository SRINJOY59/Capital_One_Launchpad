#!/bin/bash

# PRAGATI Agricultural AI Platform Startup Script
echo "🌾 Starting PRAGATI Agricultural AI Platform..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file from template."
        echo "📝 Please edit .env file with your actual API keys before continuing."
        echo "   Required API keys: GROQ, COHERE, HUGGINGFACE, PINECONE, TAVILY, GOOGLE, etc."
        read -p "Press Enter after updating .env file to continue..."
    else
        echo "❌ .env.example file not found. Please create .env manually."
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads cache logs models

# Build and start the application
echo "🏗️  Building Docker image..."
docker-compose build

echo "🚀 Starting the application..."
docker-compose up -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Application started successfully!"
    echo ""
    echo "🌟 PRAGATI Agricultural AI Platform is now running!"
    echo "📖 API Documentation: http://localhost:8000/docs"
    echo "💚 Health Check: http://localhost:8000/health"
    echo "🏠 Root Endpoint: http://localhost:8000/"
    echo ""
    echo "🔍 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
    echo ""
else
    echo "❌ Application failed to start. Check logs:"
    docker-compose logs
fi
