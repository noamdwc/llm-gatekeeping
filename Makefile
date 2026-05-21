format:
	uv run --active black .

lint:
	uv run --active black --check --diff .

test:
	uv run --active pytest

test-v:
	uv run --active python -X faulthandler -m pytest -vv -s

ci: lint test
