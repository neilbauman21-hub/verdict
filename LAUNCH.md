# Launch checklist — Verdict

GitHub repo is live and everything else is staged. Two blockers are credential
problems, not code problems.

## ✅ Done and verified

- [x] Core engine: one-pass letter-marker scoring + abstention (`verdict/core.py`)
- [x] `fit()` — data-driven signal selection + threshold fitting
- [x] Runnable benchmark reproducing every README number (`verdict/bench.py`)
- [x] Gradio demo (`verdict/demo/demo.py`) — smoke-tested, UI builds
- [x] **Both backends verified working**: MLX on Apple Silicon **and** transformers on CPU
- [x] Package builds (`twine check` PASSED on wheel + sdist)
- [x] `pip install -e .` works, imports clean
- [x] HF Space app written and smoke-tested locally (`verdict/space/app.py`)
- [x] **GitHub repo live**: https://github.com/neilbauman21-hub/verdict
- [x] All fixes committed and pushed (4 commits)

## ❌ Blocked — both tokens supplied were invalid

**PyPI** — `twine upload` returns `403 Forbidden`. The token decodes to a
*project-scoped* token bound to project id `19eb6a1d-c626-4af8-a76f-67212260e0ad`,
which is **not** `verdict-ai`. A project-scoped token cannot publish a
brand-new project. Needs an **account-scoped** API token (pypi.org → Account
settings → API tokens → "Add API token", scope: entire account).

**HuggingFace** — `api.create_repo` returns `401 Unauthorized: Invalid
username or password`, and `whoami-v2` with the same token returns 401 Invalid.
The token is not valid. Needs a fresh **write** token
(huggingface.co → Settings → Access Tokens → "New token", role: Write).

Neither failure is in our code. The local smoke test of the exact Space app
passed, so once a working token arrives it is a two-command deploy.

## □ Remaining launch steps (credential-gated)

1. `twine upload dist/*` with a valid account-scoped PyPI token → `pip install verdict-ai`
2. `HfApi().create_repo(..., repo_type="space", space_sdk="gradio")` + `upload_folder(space/)`
3. r/LocalLLaMA post — Laya was the only wave artifact on Reddit; the channel is still empty
4. PR into `multimodalart/jev-reproductions-tracker` (entry text drafted below)
5. HackerNews submission

## Tracker PR entry (drafted, needs the Space URL)

```
{ cat: 'trained', title: 'Verdict: abstention primitive for any causal LM',
  desc: 'No new weights — one-pass letter-marker scoring over a stock model, plus a
  fourth output: ABSTAIN. Measured on Qwen3.5-9B, 0.708 → 0.900 accuracy at
  threshold 0.70 while abstaining on 6 of 7 errors. Zero-shot where Laya's own
  base checkpoints score 0.362. Ships the calibration layer the wave is missing;
  honestly reports that the gain does not yet survive a 14/10 train-test split.',
  links: { gh: 'https://github.com/neilbauman21-hub/verdict',
           hf: '<pypi or space url when published>' },
  date: '2026-09-22' }
```
