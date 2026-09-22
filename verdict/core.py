"""Core engine: one-pass typed decisions + abstention over any causal LM.

No new weights. No training. It attaches to a model you already serve.

Scoring method — OPTION-MARKER scoring
--------------------------------------
Options are letter-labelled (A/B/C/...) and the answer is read off the
probability of the marker token at the final position of a single forward
pass. Multi-token options collapse to one marker, so there is no
tokenization mismatch and no length bias between long and short options.

Every question in a call is answered in ONE batched forward pass.

The Verdict primitive — ABSTENTION
---------------------------------
Each answer carries a confidence. If it falls below a threshold (fit per
question on held-out data) the layer returns ABSTAIN rather than answering
confidently and wrong.

Confidence signal selection (measured, see VERDICT_RESULTS.md)
--------------------------------------------------------------
There is no universal confidence signal. There is a universal procedure for
fitting one.

  * strong base model -> marker softmax confidence separates correct from wrong
  * weak base model    -> it does NOT; the model commits to a letter at
                          p ~= 0.99 whether or not it understood the input.
                          full-vocab entropy is what catches that case.

Set `signal="entropy"` on weak bases, or let `fit_thresholds()` pick.
"""
from __future__ import annotations

import time
from typing import Any, Mapping


def _softmax(z):
    z = z - z.max()
    e = z.exp()
    return e / e.sum()


class Verdict:
    """Typed decision layer over any causal LM. Adds nothing to the model.

    Parameters
    ----------
    model_id : str
        Any HF identifier loadable by the local backend (MLX, transformers...).
    backend : {"mlx", "transformers"}
        MLX for Apple Silicon; transformers elsewhere.
    signal : {"marker", "entropy", "auto"}
        Confidence signal for abstention. "auto" fits both and keeps the more
        separating one (see VERDICT_RESULTS.md for why this matters).
    """

    def __init__(self, model_id: str = "mlx-community/Qwen3.5-9B-MLX-4bit",
                 backend: str = "mlx", signal: str = "auto"):
        self.model_id = model_id
        self.signal = signal
        self.thresholds: dict[str, float] = {}
        self._signal_choice: dict[str, str] = {}
        self._backend_name = backend
        self._load(backend)

    # ------------------------------------------------------------------
    def _load(self, backend: str):
        t0 = time.time()
        if backend == "mlx":
            from mlx_lm import load

            self.model, self.tok = load(self.model_id)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float16, device_map="auto"
            )
        self.load_ms = (time.time() - t0) * 1000

    # ------------------------------------------------------------------
    def _encode(self, text: str) -> list[int]:
        if hasattr(self.tok, "encode"):
            return list(self.tok.encode(text))
        return list(self.tok(text).input_ids)

    def _decode_marker(self, letter: str) -> int:
        ids = self._encode(" " + letter)
        return ids[-1]

    # ------------------------------------------------------------------
    def _prompt(self, state: str, question: str, options: list[str]) -> str:
        lines = [f"State: {state}", f"Question: {question}", "Options:"]
        for i, o in enumerate(options):
            lines.append(f"  {chr(65 + i)}. {o}")
        lines.append("Answer:")
        return "\n".join(lines)

    def _state_text(self, state: Any) -> str:
        if isinstance(state, Mapping):
            return "\n".join(f"{k}: {v}" for k, v in state.items())
        return str(state)

    # ------------------------------------------------------------------
    def predict(self, state: Any, questions: Mapping[str, Mapping[str, Any]],
                thresholds: Mapping[str, float] | None = None) -> dict:
        """Answer every question in ONE batched forward pass.

        Parameters
        ----------
        state : str or dict
            The thing being decided about. Dicts are serialised to `k: v` lines.
        questions : {name: {"question": str, "options": [str], "type": str}}
            type is one of "choice" / "score" / "noul" (informational; scored
            identically).
        thresholds : {name: float}, optional
            Per-question abstention cutoffs. Overrides fitted thresholds.

        Returns
        -------
        {name: {"type", "options", "probs", "choice", "confidence",
                "entropy", "abstain", "latency_ms"}}
        """
        thr = dict(self.thresholds)
        if thresholds:
            thr.update(thresholds)
        state_text = self._state_text(state)

        prompts = [
            self._prompt(state_text, q["question"], list(q["options"]))
            for q in questions.values()
        ]
        enc = [self._encode(p) for p in prompts]
        maxlen = max(len(e) for e in enc)
        pad = self._pad_id()
        batch = self._pad_batch(enc, maxlen, pad)

        t0 = time.time()
        logits = self._forward(batch)
        fwd_ms = (time.time() - t0) * 1000

        lastpos = [len(e) - 1 for e in enc]
        out: dict[str, Any] = {}
        for i, (name, q) in enumerate(questions.items()):
            opts = list(q["options"])
            last = self._row_last(logits, i, lastpos[i])
            marks = [self._decode_marker(chr(65 + j)) for j in range(len(opts))]
            raw = [float(last[m]) for m in marks]

            probs = _softmax(self._as_array(raw))
            p = [float(x) for x in probs]
            ci = int(max(range(len(raw)), key=lambda k: raw[k]))
            conf_marker = p[ci]

            def _entropy(arr):
                pa = self._as_array(arr)
                return float(-(pa * self._log_clamped(pa)).sum())

            ent_marker = _entropy(p)

            # full-vocabulary entropy: how committed is the model, really?
            # kept on-device as an array; a 151k vocab round-tripped through a
            # python list costs ~13 s per question. do not materialise it.
            full = last - last.max()
            fp = full.exp()
            fp = fp / fp.sum()
            ent_full = float(-(fp * self._log_clamped(fp)).sum())

            cut = thr.get(name, 0.0)
            # A signal chosen during fit() wins; otherwise the "auto"
            # heuristic below decides. There is no universal confidence signal.
            chosen = self._signal_choice.get(name)
            # Heuristic signal selection for "auto". There is no universal
            # confidence signal (see bench / VERDICT_RESULTS.md): on a strong
            # base the marker softmax separates well, on a weak base it
            # saturates at ~0.99 regardless of comprehension and full-vocab
            # entropy is the only honest signal.
            #
            # Saturation signature: high marker confidence coexisting with a
            # scattered vocabulary distribution. Measured on 1.7B: correct
            # 0.980 / wrong 0.845 (sep 0.134) with entropy sep 0.284.
            saturated = conf_marker > 0.90 and ent_full < 0.35
            use_ent = (
                True if chosen == "entropy"
                else False if chosen == "confidence"
                else self.signal == "entropy" or (self.signal == "auto" and saturated)
            )
            abstain = (ent_full > cut) if use_ent else (conf_marker < cut)

            out[name] = {
                "type": q.get("type", "choice"),
                "options": opts,
                "probs": p,
                "choice": opts[ci],
                "marker": chr(65 + ci),
                "confidence": conf_marker,
                "entropy": ent_full,
                "marker_entropy": ent_marker,
                "signal": "entropy" if use_ent else "confidence",
                "abstain": abstain,
                "latency_ms": fwd_ms / len(questions),
            }
        return out

    # ------------------------------------------------------------------
    # backend adapters
    # ------------------------------------------------------------------
    def _pad_id(self) -> int:
        for name in ("eos_token_id", "pad_token_id"):
            v = getattr(self.tok, name, None)
            if isinstance(v, int):
                return v
        try:
            return self._encode("d")[0]
        except Exception:
            return 0

    def _log_clamped(self, arr, floor: float = -30.0):
        """log() that never returns -inf; works on mlx and torch alike."""
        backend = getattr(self, "_backend_name", "mlx")
        if backend == "transformers":
            import torch

            return torch.log(arr.clamp_min(1e-13))
        import mlx.core as mx

        return mx.log(mx.maximum(arr, 1e-13))

    def _as_array(self, data):
        backend = getattr(self, "_backend_name", "mlx")
        if backend == "transformers":
            import torch

            return torch.tensor(data, dtype=torch.float32)
        import mlx.core as mx

        return mx.array(data)

    def _pad_batch(self, enc, maxlen, pad):
        backend = getattr(self, "_backend_name", "mlx")
        rows = [e + [pad] * (maxlen - len(e)) for e in enc]
        if backend == "transformers":
            import torch

            return torch.tensor(rows, dtype=torch.long)
        import mlx.core as mx

        return mx.array(rows)

    def _forward(self, batch):
        backend = getattr(self, "_backend_name", "mlx")
        if backend == "transformers":
            with __import__("torch").no_grad():
                return self.model(batch).logits
        out = self.model(batch)
        return out.logits if hasattr(out, "logits") else out

    def _row_last(self, logits, i, pos):
        backend = getattr(self, "_backend_name", "mlx")
        if backend == "transformers":
            return logits[i, pos, :]
        return logits[i, pos, :]

    # ------------------------------------------------------------------
    def fit(self, labelled, penalty: float = 0.5) -> dict:
        """Calibrate on labelled data: choose the confidence signal AND the
        abstention threshold, per question.

        labelled : list of (state, questions, answers)
            answers maps question name -> ground-truth option.

        This is the step that makes abstention honest. There is no universal
        confidence signal (see bench): on a strong base the marker softmax
        separates correct from wrong; on a weak base it saturates and
        full-vocab entropy is the only usable signal. Selecting by measured
        separation beats guessing per-answer.

        Returns {question: {"signal", "threshold", "separation"}}.
        """
        recs: dict[str, list[dict]] = {}
        for state, qs, ans in labelled:
            res = self.predict(state, qs)
            for name, truth in ans.items():
                if name in res:
                    r = res[name]
                    recs.setdefault(name, []).append({
                        "conf": r["confidence"],
                        "ent": r["entropy"],
                        "correct": r["choice"] == truth,
                    })

        fitted: dict[str, dict] = {}
        grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        for name, rows in recs.items():
            def sep(key, higher_is_better):
                c = [r[key] for r in rows if r["correct"]]
                w = [r[key] for r in rows if not r["correct"]]
                if not c or not w:
                    return 0.0
                d = sum(c) / len(c) - sum(w) / len(w)
                return d if higher_is_better else -d

            conf_sep, ent_sep = sep("conf", True), sep("ent", False)
            use_ent = ent_sep > conf_sep
            field = "ent" if use_ent else "conf"

            best = (0.0, -1.0)
            for t in grid:
                cov = [r for r in rows if r[field] <= t] if use_ent \
                    else [r for r in rows if r[field] >= t]
                if not cov:
                    continue
                acc = sum(r["correct"] for r in cov) / len(cov)
                obj = acc - penalty * (1 - len(cov) / len(rows))
                if obj > best[1]:
                    best = (t, obj)

            self.thresholds[name] = best[0]
            self._signal_choice[name] = "entropy" if use_ent else "confidence"
            fitted[name] = {
                "signal": self._signal_choice[name],
                "threshold": best[0],
                "separation": round(max(conf_sep, ent_sep), 3),
            }
        return fitted

    def fit_thresholds(self, labelled, **kw) -> dict:
        """Deprecated alias for fit()."""
        return {k: v["threshold"] for k, v in self.fit(labelled, **kw).items()}