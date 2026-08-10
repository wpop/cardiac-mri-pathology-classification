.PHONY: install-dev lint format-check type-check test quality

install-dev:
	python -m pip install -e ".[dev]"

lint:
	python -m ruff check src tests scripts

format-check:
	python -m ruff format --check src tests scripts

type-check:
	python -m mypy src

test:
	python -m pytest

quality: lint format-check type-check
