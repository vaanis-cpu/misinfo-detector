.PHONY: install install-dev run run-dashboard seed test test-cov lint format typecheck clean docker-dev docker-up help

# ── default ────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Misinformation Graph Detector — available commands"
	@echo ""
	@echo "  Setup"
	@echo "    make install       Install production dependencies"
	@echo "    make install-dev   Install dev dependencies (includes linting, testing)"
	@echo ""
	@echo "  Run"
	@echo "    make seed          Seed sample data into the database"
	@echo "    make run           Start the FastAPI server"
	@echo "    make run-dashboard Start the Streamlit dashboard"
	@echo ""
	@echo "  Test"
	@echo "    make test          Run unit tests"
	@echo "    make test-cov      Run tests with HTML coverage report"
	@echo ""
	@echo "  Code quality"
	@echo "    make lint          Run ruff linter"
	@echo "    make format        Auto-format with black"
	@echo "    make typecheck     Run mypy type checker"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-dev    Start lightweight dev stack"
	@echo "    make docker-up     Start full production stack"
	@echo ""
	@echo "  Misc"
	@echo "    make clean         Remove build artifacts and caches"
	@echo ""

# ── setup ──────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# ── run ────────────────────────────────────────────────────────
seed:
	python scripts/seed_data.py

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run src/dashboard/app.py

# ── test ───────────────────────────────────────────────────────
test:
	pytest tests/unit/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# ── code quality ───────────────────────────────────────────────
lint:
	ruff check src/ tests/

format:
	black src/ tests/

typecheck:
	mypy src/

# ── docker ─────────────────────────────────────────────────────
docker-dev:
	docker-compose -f docker-compose.dev.yml up

docker-up:
	docker-compose up

# ── clean ──────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build *.egg-info
	@echo "Cleaned."
