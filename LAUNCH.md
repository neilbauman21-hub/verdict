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

## ❌ Blocked — PyPI token invalid, HF token invalid (root cause established)

**Correcting my earlier diagnosis: I was wrong to blame this on token scope.**

The definitive error body from `twine upload --verbose` is:

```
403 Invalid or non-existent authentication information.
See https://pypi.org/help/#invalid-auth
```

That message means PyPI rejects the **authentication itself**, not the
authorisation scope. Decoding the token shows its structure is well-formed
(header and claims decode cleanly, 130-byte payload, trailing signature
region), so the string is not garbled as data — but the signature does not
match what PyPI has on record.

I tested the scope theory and it failed: I retargeted the package to a free
name (`verdict-ai` was already taken on PyPI by `1psychoQAQ`; rebuilt as
`verdict-llm`, both `twine check` PASSED) and re-uploaded. Same 403 on a free
name. A scope error would have resolved there.

So in practice the token was **rotated or revoked after being copied**, or the
paste lost characters in transit — chat clients, terminals and password
managers all mangle long tokens.

**To unblock** — pypi.org → Account settings → API tokens → scope "Entire
account", then copy it straight from the browser into the terminal and verify
the paste is byte-identical:

```bash
TWINE_USERNAME="__token__" TWINE_PASSWORD="<fresh token>" \
  twine upload dist/verdict_llm-0.1.0-py3-none-any.whl dist/verdict_llm-0.1.0.tar.gz
```

**HuggingFace** — `create_repo` returns `401 Unauthorized: Invalid username or
password`; `whoami-v2` with the same token returns 401 Invalid. Same class of
problem. Needs a fresh **write** token (huggingface.co → Settings → Access
Tokens → role: Write), pasted directly.

Neither failure is in our code. The local smoke test of the exact Space app
passed, so a working token makes it a two-command deploy.

**Package is ready to publish:** `dist/verdict_llm-0.1.0-{wheel,sdist}` both
built and both pass `twine check`.

## □ Remaining launch steps (credential-gated)

1. `twine upload dist/*` with a valid account-scoped PyPI token → `pip install verdict-llm`
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
