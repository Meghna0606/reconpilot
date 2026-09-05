.PHONY: install test benchmark benchmark-multi run ui api clean
install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	python -m pytest -q

benchmark:
	python -m reconpilot benchmark --seed 42 --mode local

benchmark-multi:
	python -m reconpilot benchmark --seeds 42 43 44 45 46 --mode local

run:
	python -m reconpilot run --seed 42 --mode local

ui:
	streamlit run app.py

api:
	uvicorn api:app --host 0.0.0.0 --port 8000

clean:
	rm -rf benchmark/*.json data/*.db data/*.db-journal
