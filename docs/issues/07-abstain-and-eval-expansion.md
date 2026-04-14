# Issue 7: Abstain logic + expand eval taxonomy

**Labels:** `enhancement`, `evals`

## Context

Two related problems:

1. The system currently tries to answer every question. In a compliance setting, the correct behavior is often *"I don't know, the clauses don't say."* We have no way to express or measure that.
2. The eval set is 10 questions, all single-hop factual extraction ("what's the X period?"). Real contract Q&A includes harder shapes — multi-hop reasoning, comparative questions, and questions where the right answer is "unanswerable from this corpus."

## Proposed approach

### Abstain

- Add a `min_confidence` config: if the top retrieval score is below this, return a canonical abstain response (`answer: "Not answerable from provided clauses", citations: []`).
- Prompt the LLM to also abstain when the retrieved context doesn't cover the question.
- Track `abstain_rate` as a new eval metric.

### Eval taxonomy

Expand `eval_set.json` schema to tag each case:

```json
{
  "question": "...",
  "expected_clause_ids": [...],
  "kind": "single_hop" | "multi_hop" | "comparative" | "unanswerable",
  "expected_abstain": false
}
```

Report metrics broken down by `kind` — you will see faithfulness drop sharply on multi-hop and comparative.

## Acceptance criteria

- [ ] Abstain path implemented and unit-tested
- [ ] At least 3 `unanswerable` cases in the eval set to measure abstain behavior
- [ ] At least 3 `multi_hop` cases
- [ ] At least 2 `comparative` cases
- [ ] Eval report broken down by `kind` with separate faithfulness/precision/recall per bucket
- [ ] README "what I learned" updated with observed deltas across kinds

## Stretch

- [ ] A "trap" case: the retrieved clauses look related but don't actually answer the question. Measure whether the system abstains or hallucinates.
