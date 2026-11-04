PY ?= python3
FRONTEND_DIR := packages/frontend
BACKEND_DIR := packages/backend
ML_DIR := packages/ml-engine

.PHONY: help install test ml-test backend-test frontend-build up down verify clean

help:
	@echo "install        install python and frontend dependencies"
	@echo "test           run every test suite"
	@echo "ml-test        run the ml-engine tests"
	@echo "backend-test   run the backend tests"
	@echo "frontend-build build the Next.js dashboard"
	@echo "up             start the stack with docker compose"
	@echo "down           stop the stack"
	@echo "verify         run the end-to-end verification script"

install:
	$(PY) -m pip install -r $(BACKEND_DIR)/requirements.txt
	$(PY) -m pip install -r $(ML_DIR)/requirements.txt
	cd $(FRONTEND_DIR) && npm ci

test: ml-test backend-test

ml-test:
	$(PY) -m pytest $(ML_DIR)/tests -q

backend-test:
	$(PY) -m pytest $(BACKEND_DIR)/tests -q

frontend-build:
	cd $(FRONTEND_DIR) && npm run build

up:
	docker compose up --build -d

down:
	docker compose down

verify:
	bash scripts/verify-all.sh

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf $(FRONTEND_DIR)/.next
