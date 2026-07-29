# Eval results

Committed output from `python -m clauselens.evals`. Regenerate with:

```bash
python -m clauselens.evals --retrieval-only --split all --out results/retrieval.md   # offline
python -m clauselens.evals --split all --out results/latest.md                        # needs a key
```

- `retrieval.md` — recall@k / precision@k / MRR. Grades search alone. No API calls.
- `latest.md` — citation P/R/F1 + LLM-as-judge faithfulness. Grades the full pipeline.

Each report's HTML comment header records the timestamp, git SHA, split, and retrieval
params, so a number can be traced back to the code that produced it.

## Read these with the corpus size in mind

10 clauses at k=4 means every query retrieves 40% of the corpus — recall@4 = 1.00 is
nearly free, and precision@4 = 0.30 just says most of what k=4 drags in is padding.
These numbers say far more about citation behavior than about retrieval quality.
Growing the corpus (a CUAD subset, issue #2) is what makes them mean something.

## A finding worth keeping: the judge is not stable

The committed `latest.md` run grades case 4 (the adversarial cross-contract assignment
question) as **unfaithful** — with perfect citations. Re-running the identical case,
same inputs, `temperature=0`, the judge graded it **faithful**.

Both things are informative:

1. The failure mode the run surfaced is real: case 4 is the NDA-vs-MSA assignment pair
   built to confuse retrieval, and when it degrades, it degrades as
   *cited-correctly-but-described-loosely* — the exact failure class citation F1 cannot
   see and the judge exists to catch.
2. The judge itself is a model, and a single binary judgment on a borderline case is
   noise. `temperature=0` reduces variance; it does not eliminate it.

Mitigations, in the order I'd apply them: majority-of-3 judge votes on borderline
cases, tracking judge–human agreement on a hand-labeled slice, and persisting per-case
results across runs (issue #3) so flips are visible instead of anecdotal.
