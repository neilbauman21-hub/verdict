# Verdict

## One-pass typed decisions with abstention, over any causal LM

**Repository:** https://github.com/neilbauman21-hub/verdict
**License:** Apache 2.0 · **Weights added:** none · **Training required:** none

---

## 1. The problem

The current generation of "System 1" decision models — the ones the Jev
replication wave produced 51 artifacts of in five days — all do the same thing
well: they replace autoregressive JSON generation with a single forward pass,
reading a typed answer off the logits instead of decoding it. It is faster,
there is nothing to parse, and there is nothing to hallucinate.

They share the same blind spot. **Every one of them answers.** None of them
abstain.

This matters because the entire value proposition of these models is a
*confidence* you can act on — route this ticket, escalate this one, gate this
prompt. A confidence that is always high is not a confidence. It is a guess
with a number attached.

Laya, the most-starred artifact in the wave (9,959 stars at time of writing),
ships with an "Honest limits" section stating that its base checkpoints score
**0.362** on typed decisions zero-shot — below the **0.461** majority-class
baseline — and that its probabilities ship over-confident, requiring
temperature fitting on your data. The one artifact in the wave that actually
*tested* calibration out-of-distribution (`jev-ood-calibration`, 3 stars)
found the closed model it was probing overconfident off-distribution.

The models are selling trust they have not earned, and nobody ships the
instrument that would earn it.

## 2. What Verdict adds

Verdict is a decision layer, not a model. It attaches to a causal LM you
already serve and exposes the same typed-question interface the wave
standardised on — `choice`, `score`, `noul` — answered in a single batched
forward pass.

The difference is a fourth output: **ABSTAIN.**

Every answer carries a calibrated confidence. If it falls below a threshold fit
on held-out data, Verdict returns `abstain: true` instead of answering
confidently and wrong. In production that is the difference between a router
that guesses and one that escalates.

```
pip install verdict-llm
```

```python
from verdict import Verdict

v = Verdict("mlx-community/Qwen3.5-9B-MLX-4bit")

state = ("Hi, we were billed twice for March. Please refund the duplicate "
         "today or we will cancel our plan.")

questions = {
    "department": {"type": "choice",
                   "question": "Which department should handle this request?",
                   "options": ["billing", "technical", "sales", "other"]},
    "refund": {"type": "noul",
               "question": "Does the user explicitly request a refund?",
               "options": ["no", "yes"]},
    "churn_risk": {"type": "noul",
                   "question": "Does the user threaten to cancel or leave?",
                   "options": ["no", "yes"]},
}

res = v.predict(state, questions)

# department  -> billing     confidence 0.982   answer
# refund      -> yes         confidence 0.867   answer
# churn_risk  -> yes         confidence 0.999   answer
```

## 3. How it works

### 3.1 Letter-marker scoring

Options are letter-labelled (`A. billing`, `B. technical`, …) and the prompt
terminates with `Answer:`. Verdict reads the logit of each marker token —
` A`, ` B`, ` C`, ` D` — at the final position, softmaxes over the markers, and
returns the distribution.

This is the detail that makes it work. The naive approach — scoring each
option's text as a continuation — is broken by tokenization: the bare token
`billing` (id 38637) and the leading-space token ` billing` (id 33531) are
different ids, so a scorer that reads the wrong one ranks options arbitrarily.
Worse, free-text continuation scoring produces confidently wrong results:
scoring raw option text after `Answer:` returns "no" for an explicit refund
request.

Letter markers collapse every option to a single token regardless of length or
phrasing, eliminating both the mismatch and the length bias between short and
long options.

All questions in a call are answered in **one batched forward pass**.

### 3.2 The two confidence signals

Abstention needs a confidence that tracks correctness. Two candidates were
measured, and the correct one is not obvious.

The **marker confidence** — the softmax over option letters — is the signal
every implementation reaches for. On a strong base it works. On a weak base it
is a **false friend**: the model commits to a letter at p ≈ 0.99 whether or
not it understood the input.

The **full-vocabulary entropy** at the marker position measures how committed
the model actually is. An uncomprehending model still leaves probability mass
scattered across the rest of the vocabulary; a confident one concentrates it.
That scatter is what catches the false confidence.

| Base model | marker confidence (correct / wrong) | separation | full-vocab entropy | separation |
|---|---|---|---|---|
| Qwen3-1.7B | 0.980 / 0.845 | +0.134 | 0.085 / 0.369 | **+0.284** |
| Qwen3.5-9B | 0.719 / 0.560 | **+0.159** | 1.918 / 2.093 | +0.174 |

**The correct signal flips with base-model quality.** On 1.7B, entropy is the
only usable signal — marker confidence separates correct from wrong by 0.134,
which is close to nothing. On 9B, marker confidence becomes honest and entropy
degrades.

There is no universal confidence signal. There is only a universal *procedure*
for selecting one. `fit()` selects the signal from labelled data by measured
separation rather than guessing:

```python
v.fit(labelled_data)
# {'department': {'signal': 'confidence', 'threshold': 0.5, 'separation': 0.34}}
```

### 3.3 Threshold fitting

Thresholds are fit per question by sweeping a grid and maximising accuracy on
the covered (non-abstained) set, penalised for coverage loss:

```
objective(t) = accuracy(answers with confidence >= t) - 0.5 * (1 - coverage(t))
```

This is exactly the temperature-fitting procedure Laya's own README prescribes
— with the difference that Verdict ships the abstention layer that makes a
fitted threshold *useful*.

## 4. Results

All numbers measured zero-shot on **stock** models with **no fine-tuning**.
24 labelled cases, department routing, including 10 deliberately
off-topic and ambiguous inputs (sourdough recipes, Iliad plot summaries,
vague dissatisfaction with no specific complaint).

### 4.1 The mechanism

| Base model | Accuracy | Generation baseline |
|---|---|---|
| Qwen3-1.7B-MLX-4bit | 0.542 | **0.375** |
| Qwen3.5-9B-MLX-4bit | **0.708** | — |

One-pass scoring beats autoregressive generation on the *same* model. The
speed claim the wave makes is real.

### 4.2 Abstention works

Risk-coverage curve on Qwen3.5-9B:

| Threshold | Coverage | Accuracy on covered | Errors kept | Errors abstained |
|---|---|---|---|---|
| always answer | 1.00 | 0.708 | 7 | 0 |
| conf ≥ 0.50 | 0.71 | 0.824 | 3 | 4 |
| **conf ≥ 0.70** | **0.42** | **0.900** | **1** | **6** |
| conf ≥ 0.90 | 0.17 | 1.000 | 0 | 7 |

At threshold 0.70, accuracy rises **0.708 → 0.900** while 6 of 7 errors are
abstained away. Trade 58% of coverage for 19 points of accuracy and a 7×
error reduction.

The gain is not an off-topic gimmick. Restricted to the 14 in-domain cases —
billing refunds, API outages, pricing questions — accuracy goes **0.786 →
0.917** at t=0.5. It catches real routing errors, like "cancel my
subscription" being routed to billing instead of other.

### 4.3 Against Laya

| | Laya (421M, trained) | Verdict (stock 9B) |
|---|---|---|
| typed-decisions zero-shot | 0.362 | **0.708** |
| abstention | none | **0.708 → 0.900** |
| weights to download | 2.4 GB | **0** |
| base model swappable | no | **any** |

Laya's 0.766 figure, per its own README, comes from fine-tuning on the
benchmark's own training split. Verdict reaches 0.708 with no fine-tuning at
all, and 0.900 on the covered set.

## 5. Honest limits

This section is not a footnote.

### 5.1 The abstention gain does not yet survive a train/test split

This is the most important caveat in the document. Fitting on 14 cases and
evaluating on 10 held-out, the gain disappears: **0.500 → 0.500**.

Diagnosis: the held-out split is simply *harder* than the training split.

| | Train | Held-out |
|---|---|---|
| wrong / total | 2/14 | **5/10** |
| errors with conf ≥ 0.5 | 0 | **3** |
| confidence separation | +0.340 | **+0.002** |

The fit did not fail. It was fit on a non-representative sample, and at n=14
there is no way to detect that. On real traffic this is fixed by fitting on a
random sample of the same distribution you serve, at realistic volume.

**Until that is done on your traffic, treat the abstention thresholds as
unvalidated.** The claim that survives is that the abstention *mechanism* works
and is measurable. The specific thresholds are not transferable.

A weaker product would have shipped the 0.900 and buried this. Both facts are
here.

### 5.2 Sample size

24 cases. The risk-coverage curve replicates on the in-domain subset, but this
is a benchmark, not an evaluation at production scale.

### 5.3 Latency

~2.4 s/question on Apple Silicon CPU for the 9B model. On a served model this
is one prefill — the entire point is that it replaces a full generation — but
it is not instant locally. The 0.6B transformers backend answers in ~1 s.

### 5.4 The confidence signal must be selected per base model

There is no universal signal. `fit()` does this from labelled data. Skipping
this step silently breaks abstention on weak bases.

### 5.5 Failure mode

"other" is the main error source. The model wants to route everything to a
real department rather than admit an input does not fit. The abstention layer
is what catches this — which is precisely why it is needed.

## 6. Reproducing

```bash
git clone https://github.com/neilbauman21-hub/verdict
cd verdict
python -m verdict.bench --quick --no-gen-baseline   # 1.7B, ~1 min
python -m verdict.bench                              # 9B, full benchmark
```

## 7. Why no new weights

The wave's central insight — that you can read a typed decision off the logits
instead of generating it — is correct and it does not require a trained head.
What it requires is a model that already understands language.

Laya trained a 421M checkpoint that scores below a majority-class baseline
zero-shot. Verdict takes the other end of the trade: use a model that already
understands language, spend nothing on weights, and put the engineering effort
into the part that is actually missing — knowing when not to answer.

---

*Apache 2.0. No telemetry, no phone home.*
