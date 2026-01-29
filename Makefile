.PHONY: setup lint test train serve clean

setup:
	python -m pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

lint:
	ruff check .

test:
	pytest -q

train:
	python -m src.train

serve:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

