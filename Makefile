PYTHON ?= python
CLI ?= av-eval
OUTDIR ?= outputs
DATA ?= $(OUTDIR)/data.parquet
TRIPS ?= 200
SEED ?= 42

.PHONY: install format lint test generate-data run-eval clean

install:
	$(PYTHON) -m pip install -e .

format:
	@echo "Formatting placeholder (integrate black/ruff as needed)"

lint:
	@echo "Lint placeholder (add ruff/mypy)"

test:
	$(PYTHON) -m pytest

generate-data:
	$(CLI) generate-data --out $(DATA) --trips $(TRIPS) --seed $(SEED)

run-eval:
	$(CLI) run-eval --data $(DATA) --outdir $(OUTDIR)

clean:
	rm -rf $(OUTDIR) htmlcov .pytest_cache
