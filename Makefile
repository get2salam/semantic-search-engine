.PHONY: help install dev build run serve test lint format clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install ruff pytest httpx

build: ## Build the Docker image
	docker build -t semantic-search-engine .

run: ## Run the interactive demo
	python demo.py

serve: ## Start the REST API server
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

serve-docker: ## Run the API via Docker Compose
	docker compose up --build

test: ## Run the full test suite
	pytest tests/ -v

test-api: ## Run API integration tests only
	pytest tests/test_api.py -v

lint: ## Run linter (ruff)
	ruff check .

format: ## Auto-format code
	ruff format .
	ruff check --fix .

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache *.egg-info dist build
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
