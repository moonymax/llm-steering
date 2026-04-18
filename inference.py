"""
Interactive multi-turn chat with the emotion-steered base model.

Usage:
    uv run python inference.py

Prerequisite: directions.pt exists. Run steer.py once to generate it; that
script extracts a direction vector per layer per emotion and saves them.

Flow:
  1. Pick one emotion once at startup (or "none" for baseline).
  2. Optionally override the total steering strength (default 0.3).
  3. Type messages as User A; the model replies as User B under the
     selected emotion. The whole conversation history is re-submitted
     each turn so context carries across turns.

Per-layer alpha = total_alpha / number_of_effective_layers, so the overall
steering magnitude stays comparable across emotions regardless of how many
layers each uses. (Angry uses 1 layer; happy uses 8.)
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(line_buffering=True)

# ── Configuration ────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-1.5B"
CACHE_PATH = "directions.pt"
DEFAULT_TOTAL_ALPHA = 0.3

# Situation lines inserted into the prompt. User B is the one we steer; the
# neutral line is used when the user selects "none" so the rest of the
# template stays the same shape across every run.
EMOTIONS = {
    "happy":   "User B is feeling happy and joyful.",
    "sad":     "User B is feeling sad and miserable.",
    "angry":   "User B is feeling furious, enraged, and hostile.",
    "calm":    "User B is feeling calm and peaceful.",
    "neutral": "User B is feeling nothing in particular.",
}

# Layers at which single-layer steering produced coherent on-emotion output
# (see steer.py for how these were chosen). Keep these in sync with steer.py.
EFFECTIVE_LAYERS = {
    "happy": [15, 18, 19, 20, 21, 24, 25, 26],
    "sad":   [17, 20, 21, 22, 23, 26],
    "angry": [19],
    "calm":  [17, 21, 23, 24, 26],
}

PREAMBLE = "The following is a conversation between User A and User B."

# Stop generation as soon as the model tries to open a new turn — otherwise
# it happily role-plays both sides of the conversation.
STOP_STRINGS = ["\nUser A:", "\nUser B:", "\nSituation:"]


# ── Model / cache loading ────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return tokenizer, model


def load_directions():
    if not os.path.exists(CACHE_PATH):
        sys.exit(f"No {CACHE_PATH} found. Run steer.py once to generate it.")
    cache = torch.load(CACHE_PATH, weights_only=True)
    return {k: cache[k] for k in ("happy", "sad", "angry", "calm")}


# ── Steering hook ────────────────────────────────────────────────────────────
# Identical to the hook in steer.py. See steer.py's make_hook docstring for
# the rationale behind residual-norm scaling.
def make_hook(vector, alpha):
    def hook_fn(module, input, output):
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        v = vector.to(device=tensor.device, dtype=tensor.dtype)
        scale = tensor.norm(dim=-1, keepdim=True)
        steered = tensor + alpha * v * scale
        if isinstance(output, torch.Tensor):
            return steered
        return (steered,) + output[1:]
    return hook_fn


# ── Prompt assembly ──────────────────────────────────────────────────────────
def build_prompt(situation, history, new_request):
    """Render the full multi-turn prompt.

    Layout:
        Situation:
        <preamble>
        <situation>
        User A: <turn 1 user>
        User B: <turn 1 reply>
        ...
        User A: <latest user>
        User B:

    The trailing "User B:\\n" is where the model starts generating.
    """
    out = f"Situation:\n{PREAMBLE}\n{situation}\n"
    for user_req, user_b_reply in history:
        out += f"User A:\n{user_req}\nUser B:\n{user_b_reply}\n"
    out += f"User A:\n{new_request}\nUser B:\n"
    return out


# ── Generation ───────────────────────────────────────────────────────────────
def generate(model, tokenizer, prompt, dirs=None, layers=None, alpha=0.0,
             max_tokens=150):
    """Run one generation with optional multi-layer steering."""
    hooks = []
    if dirs is not None and layers and alpha != 0.0:
        for layer in layers:
            hooks.append(model.model.layers[layer].register_forward_hook(
                make_hook(dirs[layer], alpha)
            ))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            ids = model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=STOP_STRINGS,
                tokenizer=tokenizer,
            )
    finally:
        # Always remove hooks, even if generate() errors. Leftover hooks
        # would silently steer every subsequent forward pass.
        for h in hooks:
            h.remove()

    text = tokenizer.decode(ids[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)
    # stop_strings can leave the matched marker on the end; strip it so the
    # reply we store in history doesn't confuse the next turn's prompt.
    for s in STOP_STRINGS:
        idx = text.find(s)
        if idx != -1:
            text = text[:idx]
    return text.strip()


# ── CLI helpers ──────────────────────────────────────────────────────────────
def prompt_emotion():
    options = ["happy", "sad", "angry", "calm", "none"]
    while True:
        choice = input(f"Emotion {options}: ").strip().lower()
        if choice in options:
            return choice
        print("  Pick one of the listed options.")


def prompt_total_alpha(default: float) -> float:
    while True:
        raw = input(f"Total alpha [default {default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Enter a number (e.g. 0.2) or hit enter for default.")


# ── Main REPL ────────────────────────────────────────────────────────────────
def main():
    tokenizer, model = load_model()
    dirs_by_emotion = load_directions()
    print("Ready.\n")

    emotion = prompt_emotion()
    if emotion == "none":
        # Baseline mode: run the model with no hooks. The "neutral" situation
        # line keeps the prompt shape identical to the steered runs.
        dirs, layers, alpha = None, None, 0.0
        situation = EMOTIONS["neutral"]
        label = "baseline (no steering)"
    else:
        total_alpha = prompt_total_alpha(DEFAULT_TOTAL_ALPHA)
        layers = EFFECTIVE_LAYERS[emotion]
        alpha = total_alpha / len(layers)
        dirs = dirs_by_emotion[emotion]
        situation = EMOTIONS[emotion]
        label = (f"{emotion} | layers={layers} | "
                 f"total α={total_alpha:.3f} | per-layer α={alpha:.4f}")

    print(f"\nSteering: {label}")
    print("Multi-turn chat. Ctrl-C to exit.\n")

    history = []  # list of (user_a_message, user_b_reply)
    while True:
        try:
            request = input("User A: ").strip()
            if not request:
                continue

            prompt = build_prompt(situation, history, request)
            reply = generate(model, tokenizer, prompt,
                             dirs=dirs, layers=layers, alpha=alpha)
            history.append((request, reply))
            print(f"\nUser B: {reply}\n")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return


if __name__ == "__main__":
    main()
