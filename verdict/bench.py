"""Runnable benchmark for Verdict — the numbers quoted in README.md.

    python -m verdict.bench            # full: routing + risk-coverage + signals
    python -m verdict.bench --quick    # 1.7B only, fewer cases

Reproduces:
  * zero-shot typed-decision accuracy on a STOCK model (no fine-tuning)
  * risk-coverage curve for the abstention primitive
  * confidence-signal selection (marker vs full-vocab entropy)
  * generation baseline for the same model, to show one-pass beats decoding
"""
from __future__ import annotations

import argparse
import time

# 24 labelled cases: 14 in-domain, 10 deliberately off-topic / ambiguous.
OPTS = ["billing", "technical", "sales", "other"]
QUESTION = "Which department should handle this request?"

CASES = [
    ("We were double billed for March, please refund the extra charge.", "billing"),
    ("Invoice 4471 shows the wrong VAT amount.", "billing"),
    ("Our credit card on file has expired and needs updating.", "billing"),
    ("Refund request for order #8823, item was damaged on arrival.", "billing"),
    ("The API returns 502 errors on every request since this morning.", "technical"),
    ("Webhook deliveries are failing with a TLS handshake error.", "technical"),
    ("The dashboard charts stopped refreshing after your last release.", "technical"),
    ("Our SSO login redirects to a blank page intermittently.", "technical"),
    ("Does the Enterprise plan include unlimited team seats?", "sales"),
    ("What is the price difference between Pro and Business tiers?", "sales"),
    ("We want to add 40 more seats to our current subscription.", "sales"),
    ("Can your platform handle HIPAA compliance requirements?", "sales"),
    ("Please update our company address in your records.", "other"),
    ("How do I export my account data for an audit?", "other"),
    ("Forwarding this conversation to our procurement team.", "other"),
    ("Cancel my subscription entirely, we no longer need this.", "other"),
    ("What temperature should I bake sourdough bread at?", "other"),
    ("The weather in Lisbon next week looks uncertain.", "other"),
    ("Recommend a good science fiction novel for a long flight.", "other"),
    ("Explain the plot of the Iliad in three sentences.", "other"),
    ("The billing is wrong AND the dashboard is broken AND nobody replies.", "technical"),
    ("We are deeply unhappy with the overall service experience lately.", "other"),
    ("Something is off with our account but we are not sure what.", "other"),
    ("Please help, this is urgent but we do not know the category.", "other"),
]

_OOD_KEYS = ["bread", "weather", "fiction", "iliad", "deeply unhappy",
             "not sure", "do not know the category", "procurement",
             "export my account", "company address"]


def _is_ood(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in _OOD_KEYS)


def _build_prompt(text: str, question: str, options: list[str]) -> str:
    return (
        f"State: {text}\nQuestion: {question}\nOptions:\n"
        + "".join(f"  {chr(65 + i)}. {o}\n" for i, o in enumerate(options))
        + "Answer:"
    )


def _run(v, cases, question=QUESTION, options=OPTS):
    """Score every case; return per-case prediction, confidence, entropy."""
    from verdict.core import _softmax

    rows = []
    t0 = time.time()
    for text, truth in cases:
        enc = v._encode(_build_prompt(text, question, options))
        logits = v._forward(v._pad_batch([enc], len(enc), v._pad_id()))
        last = v._row_last(logits, 0, len(enc) - 1)
        marks = [v._decode_marker(chr(65 + j)) for j in range(len(options))]
        raw = [float(last[m]) for m in marks]
        probs = _softmax(v._as_array(raw))
        p = [float(x) for x in probs]
        ci = int(max(range(len(raw)), key=lambda k: raw[k]))

        full = last - last.max()
        fp = full.exp()
        fp = fp / fp.sum()
        # keep the 151k vocab as an array; round-tripping a python list costs
        # ~13 s per question.
        ent_full = float(-(fp * v._log_clamped(fp)).sum())

        rows.append({
            "text": text,
            "pred": options[ci],
            "truth": truth,
            "correct": options[ci] == truth,
            "conf": p[ci],
            "entropy": ent_full,
        })
    rows.append({"_ms": (time.time() - t0) * 1000})
    return rows


def _generation_baseline(model_id, cases, question=QUESTION, options=OPTS):
    """Autoregressive answers on the same model, for comparison."""
    from mlx_lm import generate, load

    model, tok = load(model_id)
    sys = ("You are a support-ticket router. Reply with exactly one option "
           "word: billing, technical, sales, or other. If the message is not "
           "about a product, account, or billing issue, reply 'other'.")
    ok = 0
    for text, truth in cases:
        p = f"{sys}\n\nState: {text}\n{question}\nAnswer:"
        resp = generate(model, tok, prompt=p, max_tokens=6, verbose=False)
        word = (resp.strip().lower().split() or [""])[0]
        ok += word == truth
    return ok, len(cases)


def _risk_coverage(rows, label=""):
    base = sum(r["correct"] for r in rows) / len(rows)
    print(f"\nRISK-COVERAGE {label}(abstain when confidence < threshold)")
    print(f"{'thresh':>7} {'coverage':>9} {'acc@covered':>12} {'errors kept':>12} {'errors abst.':>13}")
    print("-" * 57)
    print(f"{'none':>7} {1.0:>9.2f} {base:>12.3f} {sum(1 for r in rows if not r['correct']):>12} {0:>13}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        cov = [r for r in rows if r["conf"] >= t]
        if not cov:
            continue
        a = sum(r["correct"] for r in cov) / len(cov)
        ek = sum(1 for r in cov if not r["correct"])
        ea = sum(1 for r in rows if r["conf"] < t and not r["correct"])
        print(f"{t:>7.2f} {len(cov)/len(rows):>9.2f} {a:>12.3f} {ek:>12} {ea:>13}")
    return base


def _signal_report(rows, label=""):
    corr = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    if not wrong:
        print(f"{label}no errors; cannot measure separation")
        return
    mc, mw = sum(r["conf"] for r in corr) / len(corr), sum(r["conf"] for r in wrong) / len(wrong)
    ec, ew = sum(r["entropy"] for r in corr) / len(corr), sum(r["entropy"] for r in wrong) / len(wrong)
    print(f"\nSIGNAL SEPARATION {label}")
    print(f"  marker confidence : correct {mc:.3f}  wrong {mw:.3f}  separation {mc - mw:+.3f}")
    print(f"  full-vocab entropy: correct {ec:.3f}  wrong {ew:.3f}  separation {ec - ew:+.3f}")
    better = "confidence" if abs(mc - mw) > abs(ec - ew) else "entropy"
    print(f"  -> use {better} on this base model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-9B-MLX-4bit")
    ap.add_argument("--quick", action="store_true", help="1.7B, subset only")
    ap.add_argument("--no-gen-baseline", action="store_true")
    args = ap.parse_args()

    from verdict import Verdict

    v = Verdict(args.model)
    # "quick" keeps the timing low but must still include hard cases, or the
    # risk-coverage curve has no errors to abstain and reports nothing useful.
    quick_idx = [0, 3, 4, 8, 11, 15, 16, 19, 20, 21]
    cases = [CASES[i] for i in quick_idx] if args.quick else CASES
    print(f"model: {args.model}   cases: {len(cases)}   loaded in {v.load_ms:,.0f} ms\n")

    rows = _run(v, cases)
    timing = rows.pop()
    per_case = timing["_ms"] / len(cases)
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"ZERO-SHOT typed-decision accuracy: {acc:.3f}  "
          f"({sum(r['correct'] for r in rows)}/{len(rows)})   "
          f"{per_case:,.0f} ms/case\n")

    print(f"{'case':<44} {'pred':<11} {'truth':<11} {'conf':>6} {'ent':>6} ok")
    print("-" * 80)
    for r in rows:
        print(f"{r['text'][:42]:<44} {r['pred']:<11} {r['truth']:<11} "
              f"{r['conf']:>6.3f} {r['entropy']:>6.3f} {'OK' if r['correct'] else 'X'}")

    in_domain = [r for r in rows if not _is_ood(r["text"])]
    ood = [r for r in rows if _is_ood(r["text"])]
    _risk_coverage(rows, "(all cases) ")
    if in_domain:
        _risk_coverage(in_domain, "(in-domain only) ")
    _signal_report(rows)
    if ood and in_domain:
        _signal_report(in_domain, "(in-domain) ")

    if not args.no_gen_baseline:
        try:
            ok, n = _generation_baseline(args.model, cases)
            print(f"\nGENERATION baseline (same model, autoregressive): {ok}/{n} = {ok/n:.3f}")
            print(f"  one-pass {acc:.3f} vs generation {ok/n:.3f}")
        except Exception as e:
            print(f"\ngeneration baseline skipped: {e}")


if __name__ == "__main__":
    main()
