.PHONY: demo eval eval-offline test check clean

## Index from the committed cache and serve the playground. No API key needed to boot;
## /ask still requires one.
demo:
	python -m clauselens.seed --offline
	uvicorn clauselens.app:app

## Full eval on the dev split. Requires OPENAI_API_KEY.
eval:
	python -m clauselens.evals --split dev --out results/latest.md

## Retrieval-only eval across every case. Runs offline.
eval-offline:
	python -m clauselens.seed --offline
	python -m clauselens.evals --retrieval-only --split all --out results/retrieval.md

## Everything that runs without a key.
test:
	pytest tests/test_metrics.py tests/test_store.py tests/test_embed_cache.py \
	       tests/test_prompt_contract.py tests/test_retrieval_offline.py -v

## What CI enforces.
check:
	ruff format --check .
	ruff check .
	mypy clauselens/ --ignore-missing-imports
	$(MAKE) test

clean:
	rm -f clauselens.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
