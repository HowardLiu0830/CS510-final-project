.PHONY: env run test eval lint format clean

env:
	conda env create -f environment.yml

run:
	streamlit run app/streamlit_app.py

test:
	pytest -q

eval:
	python scripts/run_eval.py --input data/eval/sample_results.jsonl --output data/eval/runs/latest.json

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
