.PHONY: setup test demo bench clean

setup:
	pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
	pip3 install --break-system-packages -r requirements.txt

test:
	python3 -m pytest tests/ -q

demo:
	python3 scripts/demo.py

bench:
	python3 scripts/bench.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache
