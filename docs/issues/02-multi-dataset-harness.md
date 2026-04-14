# Issue 2: Multi-dataset harness

**Labels:** `enhancement`, `evals`, `infra`

## Context

Today there's one corpus (`data/sample_clauses.json`) and one eval set (`data/eval_set.json`). You can't ask "how does the system perform on NDA questions vs. MSA questions?" — you have to physically swap files.

To iterate meaningfully on retrieval and prompt quality, we need to run the same pipeline across multiple labeled datasets and compare.

## Proposed layout

```
data/datasets/
  toy_v0/
    clauses.json
    eval_set.json
    README.md       # what's in this dataset, how it was sourced
  cuad_liability/
    ...
  cuad_termination/
    ...
  nda_subset/
    ...
```

## Acceptance criteria

- [ ] `--dataset <name>` CLI flag on the eval runner
- [ ] Loader utility that reads `data/datasets/<name>/{clauses,eval_set}.json`
- [ ] `seed.py --dataset <name>` scopes the DB table per dataset (or uses separate DB files)
- [ ] At least 2 datasets checked in: `toy_v0` (current) + one CUAD subset
- [ ] README updated with `python -m clauselens.evals --dataset cuad_liability` example

## Stretch

- [ ] `--dataset all` runs every dataset and prints a comparison table
