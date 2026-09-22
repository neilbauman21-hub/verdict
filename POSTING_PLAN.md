# Where to post Verdict

Grounded in the actual announcement channels the 51-artifact wave used, not
guessing. Reddit is the key insight: **1 of 51 artifacts posted there, and it
was the most-starred one by 3×.** The channel won Laya the race and is still
nearly empty.

---

## 1. Primary — Reddit (highest expected value)

**r/LocalLLaMA** — **the target.** ~700k subscribers, self-hosting
practitioners, the exact audience for "runs on a model you already serve."
Laya was the *only* wave artifact to post here and it 3×'d every competitor
that crowded X instead. The post is written for this sub specifically.

**r/ollama** (~250k) — the same audience from the serving angle. Reframe the
title to the deployment benefit: "one-pass decisions on the model you already
run."

**r/LLMDevs** / **r/llmengineering** (~30-50k each) — smaller, but the builder
audience. Post with the technical framing; the letter-marker bug and the
entropy performance fix play well here.

**r/MachineLearning** — ~1.3M, but strict moderation and low tolerance for
self-promotion. Post only if the r/LocalLLaMA post performs, and lead with the
confidence-signal finding (it reads as a research result, not a product pitch).
Check the rules first — this sub removes promotional content.

### Reddit posting notes
- Post between **9am–2pm ET on a weekday** for the strongest window
- The first 10 votes determine whether it surfaces. Do not post and vanish —
  be present in the comments for the first two hours
- Title is load-bearing. "I built X" underperforms "the finding"; the current
  title leads with the gap (0 of 51 abstain) rather than the product
- **Do not crosspost to all four subs at once.** Reddit's spam filters treat
  identical submissions across subs as promotion. Post r/LocalLLaMA first,
  wait 24h, then adapt per sub

---

## 2. HackerNews

**news.ycombinator.com** — the tracker research shows 0 of 51 wave artifacts
used it, and this story has the shape HN rewards: a technical finding
(the confidence-signal flip) plus visible honesty (publishing your own failed
train/test split).

- Title: `Show HN: A decision layer that abstains` or lead with the finding:
  `Confidence signals flip with model size (in decision LMs)`
- Post **8–10am ET**. Avoid Friday afternoon and weekends
- The comments will test the honest-limits claim. That is the point — answer
  it directly, it's the strongest thing in the doc

---

## 3. The wave's own aggregator (do not skip)

**`multimodalart/jev-reproductions-tracker`** — 92 likes, the canonical grid
the whole wave reads. Laya is entry #30 there. This is where attention in the
wave actually flows; getting listed next to Laya is worth more than a post.

Open a PR adding an entry to the `ITEMS` array in `index.html`:

```js
{ cat: 'trained', title: 'Verdict: abstention primitive for any causal LM',
  by: { name: 'neilthepineapplegoat', handle: null },
  desc: 'No new weights — one-pass letter-marker scoring over a stock causal LM, '
        'plus a fourth output: ABSTAIN. Measured on Qwen3.5-9B: 0.708 zero-shot '
        '(vs Laya base 0.362), 0.708 -> 0.900 at threshold 0.70 while abstaining '
        'on 6 of 7 errors. Ships the calibration layer the wave is missing; '
        'reports honestly that the gain does not yet survive a 14/10 train-test split.',
  links: { gh: 'https://github.com/neilbauman21-hub/verdict' },
  date: '2026-09-22', tags: 'abstention calibration marker-scoring routing guardrails' },
```

Add the PyPI URL to `links` once published.

---

## 4. Secondary — where the wave actually is (crowded)

**X/Twitter** — 30 of 51 artifacts used it. High reach, but you're the 51st
fish in a crowded feed. Post only *after* the Reddit post is up, and link to
the Reddit post rather than duplicating content — this routes the audience to
the less-crowded channel and signals discussion is already happening.

**dev.to** — 1 of 51 artifacts, and it's where Laya's write-up lives (48
reactions, modest). A technical write-up of the confidence-signal finding fits
the platform better than a product announcement. This is the right home for
the TECHNICAL.md content with the bug-fixes section expanded.

---

## 5. Sequence (order matters)

1. **GitHub README + TECHNICAL.md live** (done)
2. **PyPI** `pip install verdict-llm` — blocked on a valid account-scoped token.
   Laya's single biggest differentiator was being the only `pip install` in the
   wave. Do not post anywhere until this resolves, or the README's install
   command is broken
3. **HF Space** — blocked on a valid HF write token. Try-before-install is
   worth a lot; skip if it takes more than a day and revisit
4. **r/LocalLLaMA** — the main event
5. **Tracker PR** — same day, ideally a few hours after the post has votes
6. **HN** — next morning, 8–10am ET
7. **X + dev.to** — 24h after Reddit, linking back to it

---

## What not to do

- **Don't post to all subreddits simultaneously** — spam-filter risk, and it
  reads as promotion
- **Don't post before PyPI resolves** — the README says `pip install
  verdict-llm` and a broken install command kills the launch
- **Don't soften the honest-limits section for the post** — the failed
  train/test split is the most credible thing in it. Laya got 9,959 stars with
  an "Honest limits" section; the wave rewards it
- **Don't claim 0.900 without the split caveat** — they're both in the same
  document and the inconsistency would be the first thing the comments find
