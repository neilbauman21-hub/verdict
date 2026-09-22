# Verdict

**Typed decisions with abstention, over any causal LM.**

Give it a **state** (text, email, ticket, or JSON) and **typed questions**; it returns typed answers with calibrated probabilities in **one forward pass** — no text generation, nothing to parse, nothing to hallucinate. And when it isn't sure, it says so.

```bash
pip install verdict-ai
```

```python
from verdict import Verdict

v = Verdict("mlx-community/Qwen3.5-9B-MLX-4bit")

state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."

questions = {
    "department": {"type": "choice", "question": "Which department should handle this request?",
                   "options": ["billing", "technical", "sales", "other"]},
    "refund":     {"type": "noul",   "question": "Does the user explicitly request a refund?",
                   "options": ["no", "yes"]},
    "churn_risk": {"type": "noul",   "question": "Does the user threaten to cancel or leave?",
                   "options": ["no", "yes"]},
}

res = v.predict(state, questions)
for name, r in res.items():
    print(f"{name:<11} -> {r['choice']:<12} confidence {r['confidence']:.3f}  {'ABSTAIN' if r['abstain'] else 'answer'}")
```

Every question in a call is answered in a **single batched forward pass**.

---

## The primitive nobody else ships: ABSTENTION

Every decision model in the current wave answers. **None of them say "I don't know."** That is the hole Verdict fills.

Each answer carries a calibrated confidence. If it falls below a threshold fit on your data, Verdict returns `ABSTAIN` instead of a confident wrong answer:

| Threshold | Coverage | Accuracy on covered | Errors kept | Errors abstained |
|---|---|---|---|---|
| always answer | 1.00 | 0.708 | 7 | 0 |
| conf ≥ 0.50 | 0.71 | 0.824 | 3 | 4 |
| **conf ≥ 0.70** | **0.42** | **0.900** | **1** | **6** |
| conf ≥ 0.90 | 0.17 | 1.000 | 0 | 7 |

Measured zero-shot on a **stock** Qwen3.5-9B with **no fine-tuning**. Trade 58% of coverage for +19 points of accuracy and a 7× error reduction. For any router that can escalate to a human or a frontier model, that trade is obviously correct.

```python
v.fit_thresholds(labelled_data)     # one line, on your own traffic
res["department"]["abstain"]        # -> True  (escalate instead of guessing)
```

---

## Why no new weights

Laya's own benchmarks show its base checkpoints score **0.362** on typed decisions zero-shot — below a **0.461** majority-class baseline — and that the 0.766 figure comes from fine-tuning on the benchmark's own training split. Its 1,721 Hub likes sit alongside **zero downloads**.

Verdict takes the other end of the trade: **use a model that already understands language.** No trained head, no checkpoint to download, no fine-tuning loop. Attach it to whatever you already serve.

| | Laya (421M, trained) | Verdict (stock 9B) |
|---|---|---|
| typed-decisions zero-shot | 0.362 | **0.708** |
| abstention | none | **0.708 → 0.900** |
| weights to download | 2.4 GB | **0** |
| base model swappable | no | **any** |

One-pass scoring also beats generation on the *same* model: **0.542 vs 0.375** on Qwen3-1.7B. The speed claim the wave makes is real.

---

## The confidence signal is not obvious — read this

There is **no universal confidence signal.** There is a universal *procedure* for fitting one, and choosing wrong silently breaks abstention.

| Base model | marker confidence | full-vocab entropy |
|---|---|---|
| Qwen3-1.7B (weak) | correct 0.980 / wrong 0.845 → sep **0.134** | correct 0.085 / wrong 0.369 → sep **0.284** |
| Qwen3.5-9B (strong) | correct 0.719 / wrong 0.560 → sep **0.159** | correct 1.918 / wrong 2.093 → sep 0.174 |

**The signal flips with base-model quality.** On a weak base the marker softmax is a false friend — the model commits to a letter at p≈0.99 whether or not it understood you. Full-vocab entropy catches that, because an uncomprehending model leaves probability mass scattered across the vocabulary. On a strong base the marker confidence becomes honest and entropy degrades.

`signal="auto"` measures both and picks the separating one. On weak bases it switches to entropy.

---

## Honest limits

- **24 cases in the published run.** The risk-coverage curve holds and replicates on the in-domain subset (0.733 → 0.917), but the thresholds need refitting on real traffic before production use.
- **9B inference is ~2.4 s/question on Apple Silicon CPU.** On a served model this is one prefill — the whole point is it replaces a full generation — but it is not instant locally.
- **"other" is the main error source.** The model wants to route everything to a real department rather than admit it doesn't fit. The abstention layer is what catches this.
- **Not a replacement for a frontier model on hard reasoning.** It is a fast, honest gate in front of one.

---

## Benchmarks

Reproduce the numbers above:

```bash
git clone https://github.com/neilthepineapplegoat/verdict
cd verdict
python -m verdict.bench        # routing + risk-coverage + signal selection
```

Apache 2.0. No telemetry, no phone home.
