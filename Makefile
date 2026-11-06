.PHONY: install index eval serve test lint docker

install:
	pip install -e ".[api,pdf,dev]"

index:
	python -m sanad index

eval:
	python -m sanad eval --out eval/results.md

serve:
	uvicorn sanad.api:app --reload

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

docker:
	docker compose up --build
