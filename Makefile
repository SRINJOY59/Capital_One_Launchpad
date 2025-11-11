.PHONY: help build up down logs clean dev test

# Default target
help:
	@echo "🌾 PRAGATI Agricultural AI Platform - Docker Commands"
	@echo ""
	@echo "Production Commands:"
	@echo "  make build     - Build the Docker image"
	@echo "  make up        - Start the application in production mode"
	@echo "  make down      - Stop the application"
	@echo "  make logs      - View application logs"
	@echo "  make restart   - Restart the application"
	@echo ""
	@echo "Development Commands:"
	@echo "  make dev       - Start in development mode with hot reload"
	@echo "  make dev-down  - Stop development environment"
	@echo "  make dev-logs  - View development logs"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean     - Clean Docker images and containers"
	@echo "  make test      - Run health check test"
	@echo "  make shell     - Open shell in running container"
	@echo ""

# Production commands
build:
	@echo "🏗️  Building PRAGATI Agricultural AI Platform..."
	docker-compose build

up:
	@echo "🚀 Starting PRAGATI Agricultural AI Platform..."
	docker-compose up -d
	@echo "✅ Application started! Check http://localhost:8000/docs"

down:
	@echo "🛑 Stopping PRAGATI Agricultural AI Platform..."
	docker-compose down

logs:
	@echo "📋 Viewing application logs..."
	docker-compose logs -f

restart: down up

# Development commands
dev:
	@echo "🔧 Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Development environment started! Check http://localhost:8000/docs"

dev-down:
	@echo "🛑 Stopping development environment..."
	docker-compose -f docker-compose.dev.yml down

dev-logs:
	@echo "📋 Viewing development logs..."
	docker-compose -f docker-compose.dev.yml logs -f

# Utility commands
clean:
	@echo "🧹 Cleaning Docker images and containers..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

test:
	@echo "🧪 Testing application health..."
	@if curl -f http://localhost:8000/health > /dev/null 2>&1; then \
		echo "✅ Application is healthy!"; \
	else \
		echo "❌ Application is not responding"; \
		exit 1; \
	fi

shell:
	@echo "🐚 Opening shell in agricultural-ai container..."
	docker exec -it agricultural-ai-app /bin/bash

# Setup commands
setup:
	@echo "⚙️  Setting up PRAGATI Agricultural AI Platform..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "📝 Created .env file. Please update with your API keys."; \
	fi
	@mkdir -p uploads cache logs models
	@echo "✅ Setup complete!"

# Quick start
start: setup build up test
	@echo "🌟 PRAGATI Agricultural AI Platform is ready!"
	@echo "📖 API Documentation: http://localhost:8000/docs"
	@echo "💚 Health Check: http://localhost:8000/health"
