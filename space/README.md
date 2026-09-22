---
title: Verdict — typed decisions with abstention
emoji: ⚖️
colorFrom: slate
colorTo: indigo
sdk: gradio
sdk_version: "5.9.1"
app_file: demo.py
pinned: false
license: apache-2.0
short_description: One-pass typed decisions; the model says ABSTAIN when it isn't sure
---

# Verdict — typed decisions with abstention

Give it a **state** (text, email, ticket) and **typed questions**; it returns
typed answers with calibrated probabilities in **one forward pass** — no text
generation, nothing to parse, nothing to hallucinate.

The primitive the current wave of decision models is missing: **ABSTENTION.**
Every other model answers. Verdict is the one that knows when not to.

- No new weights, no training — works on any causal LM
- All questions in a call answered in a single batched forward pass
- Each answer carries a calibrated confidence; below a fitted threshold it
  returns **ABSTAIN** instead of a confident wrong answer

The demo runs on a small stock model so it stays free to host. Bring your own
model in the box on the left.

Source: https://github.com/neilbauman21-hub/verdict
