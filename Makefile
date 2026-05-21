.PHONY: install-deps repro format lint test test-v ci

install-deps:
	conda env update -f environment.yml

repro:
	uv run --active --no-project dvc repro

format:
	uv run --active --no-project black .

lint:
	uv run --active --no-project black --check --diff .

test:
	uv run --active --no-project pytest

test-v:
	uv run --active --no-project python -X faulthandler -m pytest -vv -s

ci: lint test
