# Issue 3: Persist eval runs to a local results table

**Labels:** `enhancement`, `evals`, `observability`

## Context

Eval results are printed to stdout and lost. There's no way to answer "did my last prompt change actually help?" beyond eyeballing two terminal outputs.

We need an append-only log of every eval run with enough metadata to reconstruct what was tested.

## Schema

`results/runs.db` (SQLite), table `eval_runs`:

| col | type | example |
|---|---|---|
| id | INTEGER PK | 1 |
| timestamp | TEXT | 2026-04-14T21:30:11Z |
| git_sha | TEXT | 9a3f2e1 |
| dataset | TEXT | toy_v0 |
| embed_model | TEXT | text-embedding-3-small |
| chat_model | TEXT | gpt-4o-mini |
| top_k | INTEGER | 4 |
| score_threshold | REAL | 0.0 |
| n_cases | INTEGER | 10 |
| faithfulness | REAL | 0.80 |
| citation_precision | REAL | 0.55 |
| citation_recall | REAL | 0.90 |
| citation_f1 | REAL | 0.68 |
| notes | TEXT | "baseline" |

## Acceptance criteria

- [ ] Eval runner writes one row per run
- [ ] `python -m clauselens.leaderboard` prints the last N runs sorted by F1
- [ ] `--notes "prompt v2"` flag to tag runs
- [ ] Git SHA auto-captured (dirty working tree shown as `9a3f2e1-dirty`)

## Stretch

- [ ] Per-case rows in a second table so you can diff which cases regressed between two runs
