**Every decision model in the Jev wave answers. None of them say "I don't know." I built the one that does.**

There are 51 open reproductions of TypeSafe's Jev right now. They all do the same good thing: replace JSON generation with a single forward pass, reading the answer off the logits. Faster, nothing to parse, nothing to hallucinate.

They all have the same blind spot. All 51 of them answer. Zero ship an abstention primitive.

That's a problem, because the entire pitch of these models is a *confidence* you can act on — route this ticket, escalate that one, gate this prompt. A confidence that's always high isn't a confidence. It's a guess with a number attached.

**Verdict** is a decision layer with a fourth output: ABSTAIN. It attaches to any causal LM you already serve. No new weights, no training, nothing to download.

    pip install verdict-llm

Give it a state and typed questions, it answers all of them in one forward pass. If the calibrated confidence falls below a threshold you fit on your own data, it abstains and escalates instead of guessing.

**The results** — all zero-shot, no fine-tuning, stock Qwen3.5-9B:

- 0.708 accuracy vs Laya's 0.362 typed-decisions zero-shot baseline (their own README admits this is below their 0.461 majority-class baseline; their 0.766 comes from fine-tuning on the benchmark's training split)
- With abstention at threshold 0.70: **0.708 → 0.900**, abstaining on 6 of 7 errors
- In-domain only: 0.786 → 0.917. Not an off-topic gimmick — it catches real routing errors like "cancel my subscription" going to billing
- One-pass scoring (0.542) beats generation (0.375) on the same 1.7B model

**The interesting part is what I found about confidence.** Your instinct is to read the softmax over the options. On a strong model that works. On a weak one it's a false friend — the model commits to a letter at p≈0.99 whether or not it understood you. I measured 0.999 confidence when correct and 0.986 when wrong. Separation of 0.013. Essentially nothing.

The thing that actually catches it is full-vocabulary entropy at the marker position — an uncomprehending model leaves probability mass scattered across the vocab, a confident one concentrates it. On 1.7B that's 4x more separating than marker confidence. On 9B, they flip. So there's no universal confidence signal, only a universal procedure for fitting one.

**Now the honest part, because I'd rather you hear it from me:**

The abstention gain does NOT survive a train/test split at my sample size. Fit on 14 cases, evaluate on 10 held-out: 0.500 → 0.500. Flat.

I dug into it. The held-out split is just harder than the training split — 2/14 wrong in train vs 5/10 held-out, and 3 of those errors carry confidence ≥ 0.5. The fit didn't fail, it was fit on a non-representative sample, and at n=14 there's no way to tell.

So the mechanism works and is measurable. The thresholds are not transferable yet. They need refitting on real traffic at realistic volume before anyone should trust them in production. That's in the README as the headline limit, not buried.

**Bugs I hit and fixed, in case you're building something similar:**

- Scoring bare option tokens vs leading-space tokens gives different ids (`billing`=38637, ` billing`=33531). Read the wrong one and your option ranking is arbitrary. Fixed by letter-marker scoring — every option becomes one A/B/C/D token
- Computing full-vocab entropy by round-tripping a 151k vocab through a Python list costs 13 seconds per question. Same computation on-device: 40s → 2.4s for three questions
- transformers 5 renamed `torch_dtype` → `dtype`. The old one doesn't just warn, it segfaults during weight loading

Repo: https://github.com/neilbauman21-hub/verdict

Benchmark reproduces every number above: `python -m verdict.bench`

Apache 2.0, no telemetry. Happy to answer questions — particularly if you have real labelled traffic to fit thresholds on, because that's the thing that would turn the honest caveat into a real result.

---

*r/LocalLLaMA — I'm posting this here instead of X because all 51 other artifacts in this wave announced on X. Nobody's posted here. Figured this crowd is exactly the people who'd actually run it.*
