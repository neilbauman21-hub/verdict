"""Space bootstrap: installs verdict from the repo, then runs the Gradio demo.

On a free CPU Space there is no MLX and no GPU, so the demo uses the
`transformers` backend on a small stock model. The interesting part — the
abstention primitive — is backend-independent.
"""
import os
import subprocess
import sys

# Install the package + the transformers backend if not already present.
try:
    import verdict  # noqa: F401
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "git+https://github.com/neilbauman21-hub/verdict",
         "transformers>=4.45", "torch", "accelerate", "gradio"]
    )

import gradio as gr  # noqa: E402

from verdict import Verdict  # noqa: E402

# Small stock model so the Space stays free to run. Bring your own in the UI.
DEFAULT_MODEL = os.environ.get("VERDICT_MODEL", "Qwen/Qwen3-0.6B")

DEFAULT_STATE = (
    "Hi, we were billed twice for March. Please refund the duplicate today "
    "or we will cancel our plan."
)

DEFAULT_QUESTIONS = """{
  "department": {
    "type": "choice",
    "question": "Which department should handle this request?",
    "options": ["billing", "technical", "sales", "other"]
  },
  "refund": {
    "type": "noul",
    "question": "Does the user explicitly request a refund?",
    "options": ["no", "yes"]
  },
  "urgency": {
    "type": "score",
    "question": "How urgent is this request?",
    "options": ["not urgent", "somewhat urgent", "critical"]
  },
  "churn_risk": {
    "type": "noul",
    "question": "Does the user threaten to cancel or leave?",
    "options": ["no", "yes"]
  }
}"""

_v = None


def get_model(model_id):
    global _v
    if _v is None or _v.model_id != model_id:
        _v = Verdict(model_id, backend="transformers")
    return _v


def run(state, questions_json, model_id, threshold):
    import json

    try:
        qs = json.loads(questions_json)
    except Exception as e:
        return f"**Invalid question JSON:** {e}"

    try:
        v = get_model(model_id)
    except Exception as e:
        return f"**Could not load `{model_id}`:** {e}"

    try:
        res = v.predict(state, qs)
    except Exception as e:
        return f"**Prediction error:** {e}"

    lines = []
    for name, r in res.items():
        r = dict(r)
        if threshold:
            r["abstain"] = r["confidence"] < threshold
        verdict_tag = "\u26d4 **ABSTAIN** \u2014 escalate instead of guessing" if r["abstain"] else "\u2705 answer"
        lines.append(f"### `{name}` \u2014 {verdict_tag}")
        lines.append(f"**\u2192 {r['choice']}**  \u00b7  confidence **{r['confidence']:.3f}**  "
                     f"\u00b7  signal `{r['signal']}`  \u00b7  {r['latency_ms']:.0f} ms")
        lines.append("")
        for o, p in zip(r["options"], r["probs"]):
            bar = "\u2588" * int(p * 36)
            lines.append(f"`{o:<16}` {p:.3f}  {bar}")
        lines.append("")
    return "\n".join(lines)


with gr.Blocks(title="Verdict \u2014 typed decisions with abstention") as demo:
    gr.Markdown(
        "# Verdict \u2014 typed decisions with abstention\n"
        "One forward pass over any causal LM. **No generation, no new weights.**\n\n"
        "Every other decision model answers. This one says **ABSTAIN** when it isn't sure \u2014 "
        "measured: accuracy 0.708 \u2192 0.900 while abstaining on 6 of 7 errors."
    )
    with gr.Row():
        with gr.Column(scale=1):
            state = gr.Textbox(label="State", value=DEFAULT_STATE, lines=6)
            questions = gr.Code(label="Questions (JSON)", value=DEFAULT_QUESTIONS, language="json")
            model_id = gr.Textbox(label="Model (any HF causal LM)", value=DEFAULT_MODEL)
            threshold = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                  label="Manual abstention threshold (0 = always answer)")
            btn = gr.Button("Decide", variant="primary")
        with gr.Column(scale=1):
            out = gr.Markdown(label="Decisions")
    btn.click(run, [state, questions, model_id, threshold], out)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
