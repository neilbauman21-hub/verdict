"""Live demo for Verdict — typed decisions with abstention.

Run locally:
    python demo/demo.py

Deploy as a Hugging Face Space (Gradio):
    gradio deploy
"""
from __future__ import annotations

import gradio as gr

from verdict import Verdict

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
        _v = Verdict(model_id)
    return _v


def run(state, questions_json, model_id, threshold):
    import json

    try:
        qs = json.loads(questions_json)
    except Exception as e:
        return f"**Invalid question JSON:** {e}"

    v = get_model(model_id)
    try:
        res = v.predict(state, qs)
    except Exception as e:
        return f"**Prediction error:** {e}"

    lines = []
    for name, r in res.items():
        verdict_tag = "🚫 **ABSTAIN**" if r["abstain"] else "✅ answer"
        if threshold is not None and not r["abstain"]:
            # let the user's manual threshold override
            r = dict(r)
            r["abstain"] = r["confidence"] < threshold
            verdict_tag = "🚫 **ABSTAIN**" if r["abstain"] else "✅ answer"
        lines.append(f"### `{name}` — {verdict_tag}")
        lines.append(f"**→ {r['choice']}**  ·  confidence **{r['confidence']:.3f}**  "
                     f"·  signal `{r['signal']}`  ·  {r['latency_ms']:.0f} ms")
        lines.append("")
        for o, p in zip(r["options"], r["probs"]):
            bar = "█" * int(p * 36)
            lines.append(f"`{o:<16}` {p:.3f}  {bar}")
        lines.append("")
    return "\n".join(lines)


def build():
    with gr.Blocks(title="Verdict — typed decisions with abstention", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Verdict — typed decisions with abstention\n"
            "One forward pass over any causal LM. **No generation, no new weights.**\n"
            "The primitive nobody else ships: it says **ABSTAIN** when it isn't sure."
        )
        with gr.Row():
            with gr.Column(scale=1):
                state = gr.Textbox(label="State", value=DEFAULT_STATE, lines=6)
                questions = gr.Code(label="Questions (JSON)", value=DEFAULT_QUESTIONS, language="json")
                model_id = gr.Textbox(label="Model", value="mlx-community/Qwen3.5-9B-MLX-4bit")
                threshold = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                      label="Manual abstention threshold (0 = always answer)")
                btn = gr.Button("Decide", variant="primary")
            with gr.Column(scale=1):
                out = gr.Markdown(label="Decisions")
        btn.click(run, [state, questions, model_id, threshold], out)
    return demo


if __name__ == "__main__":
    build().launch()
