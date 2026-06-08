.PHONY: help install test test-watch lint format check analyse report serve clean-stems clean-features clean-reports clean verify

# Default target
help:
	@echo "KLF Manual Analyser"
	@echo ""
	@echo "Setup"
	@echo "  make install          Install all dependencies via uv"
	@echo "  make verify           Verify all libraries import correctly"
	@echo ""
	@echo "Development"
	@echo "  make test             Run all tests"
	@echo "  make test-watch       Run tests in watch mode (requires pytest-watch)"
	@echo "  make lint             Check code with ruff"
	@echo "  make format           Format code with ruff"
	@echo "  make check            lint + test"
	@echo ""
	@echo "Pipeline"
	@echo "  make analyse          Analyse tracks (MODE and PATH required)"
	@echo "                        make analyse MODE=1988 PATH=./my-tracks"
	@echo "  make report           Re-render HTML report from DB"
	@echo "                        make report MODE=1988"
	@echo "  make serve            Start local report server (port 8000)"
	@echo ""
	@echo "Cache"
	@echo "  make clean-stems      Remove WAV stems from data/stems/"
	@echo "  make clean-features   Remove all track records from DB"
	@echo "  make clean-reports    Remove rendered HTML reports"
	@echo "  make clean            Remove everything (stems + DB + reports)"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:
	uv sync

verify:
	uv run python scripts/verify_libs.py

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

test:
	uv run pytest -v

test-watch:
	uv run pytest-watch -- -v

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

check: lint test

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Usage: make analyse MODE=1988 PATH=./my-tracks
analyse:
ifndef MODE
	$(error MODE is required. Usage: make analyse MODE=1988 PATH=./my-tracks)
endif
ifndef PATH
	$(error PATH is required. Usage: make analyse MODE=1988 PATH=./my-tracks)
endif
	uv run manual-analyser analyse $(PATH) --mode $(MODE)

# Usage: make report MODE=1988
report:
ifndef MODE
	$(error MODE is required. Usage: make report MODE=1988)
endif
	uv run manual-analyser report --mode $(MODE)

serve:
	uv run manual-analyser serve

# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

clean-stems:
	uv run manual-analyser clean --stems

clean-features:
	uv run manual-analyser clean --features

clean-reports:
	uv run manual-analyser clean --reports

clean:
	uv run manual-analyser clean

clean-py:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete