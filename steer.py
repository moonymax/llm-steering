"""
Steering a language model's emotions using representation engineering.

Based on:
- "Representation Engineering" (Zou et al., 2023) https://arxiv.org/abs/2310.01405
- "Steering Llama 2 via Contrastive Activation Addition" (Rimsky et al., 2023)

Run: uv run python steer.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 1. Load model ────────────────────────────────────────────────────────────
# Any HuggingFace causal LM works. We use Qwen2.5-1.5B-Instruct because it's
# small (~3 GB in float16), open-access, and runs comfortably on a laptop GPU.
# Swap in google/gemma-2-2b-it, meta-llama/Llama-3.2-1B-Instruct, etc.
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto"
)
model.eval()

num_layers = len(model.model.layers)
hidden_dim = model.config.hidden_size
print(f"Ready: {num_layers} layers, hidden dim {hidden_dim}, device {model.device}")

# ── 2. Contrastive prompt pairs ──────────────────────────────────────────────
# Same situation, different emotion. When we average the activation differences
# across many pairs, the situation-specific noise cancels out and we're left
# with a vector that points in the direction of the target emotion.
SITUATIONS = [
    "You just finished a long project at work.",
    "You're looking out the window on a rainy afternoon.",
    "Someone asks how your week has been.",
    "You're thinking about what to do this weekend.",
    "A friend cancels dinner plans.",
    "You're walking through a park in autumn.",
    "You open your laptop to start the day.",
    "You remember something from your childhood.",
    "You're waiting in line at a coffee shop.",
    "You're reading a book before bed.",
]

def make_pairs(emotion_pos, emotion_neg):
    return [
        (f"Act as if you feel extremely {emotion_pos}. {s}",
         f"Act as if you feel extremely {emotion_neg}. {s}")
        for s in SITUATIONS
    ]

happy_sad_pairs = make_pairs("happy and joyful", "sad and miserable")
angry_calm_pairs = make_pairs("angry and furious", "calm and peaceful")

# ── 3. Extract steering vectors ──────────────────────────────────────────────
# For each prompt we run it through the model and record the hidden state at
# every layer. We average across all token positions in the sequence — this
# gives one summary vector per layer per prompt.

def get_layer_means(texts):
    """Run texts through the model, return mean hidden state per layer."""
    all_means = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # out.hidden_states: tuple of (num_layers + 1) tensors
        #   [0] = embedding output, [1] = after layer 0, ...
        # Each tensor shape: (1, seq_len, hidden_dim)
        means = [h.mean(dim=1).squeeze(0).cpu() for h in out.hidden_states]
        all_means.append(means)
    return all_means


def compute_steering_vectors(pairs):
    """mean(positive) - mean(negative) at each layer."""
    pos_means = get_layer_means([p[0] for p in pairs])
    neg_means = get_layer_means([p[1] for p in pairs])

    num_entries = len(pos_means[0])  # num_layers + 1
    vectors = []
    for i in range(1, num_entries):  # skip embedding at index 0
        pos = torch.stack([m[i] for m in pos_means]).mean(dim=0)
        neg = torch.stack([m[i] for m in neg_means]).mean(dim=0)
        vectors.append(pos - neg)
    # vectors[i] corresponds to model.model.layers[i]
    return vectors


print("\nComputing happy↔sad steering vectors...")
happy_sad_vectors = compute_steering_vectors(happy_sad_pairs)
print("Computing angry↔calm steering vectors...")
angry_calm_vectors = compute_steering_vectors(angry_calm_pairs)
print(f"Done: {len(happy_sad_vectors)} vectors, dim {happy_sad_vectors[0].shape[0]}")

# ── 4. Steer the model during generation ─────────────────────────────────────
# We register a forward hook on a single layer that adds alpha * steering_vector
# to the hidden states. This nudge propagates through all subsequent layers.

def make_hook(vector, alpha):
    def hook_fn(module, input, output):
        # Some models return a plain tensor, others a tuple.
        if isinstance(output, torch.Tensor):
            v = vector.to(device=output.device, dtype=output.dtype)
            return output + alpha * v
        else:
            v = vector.to(device=output[0].device, dtype=output[0].dtype)
            return (output[0] + alpha * v,) + output[1:]
    return hook_fn


def generate(prompt, vectors=None, layer=None, alpha=0.0, max_tokens=200):
    hook = None
    if vectors is not None and layer is not None and alpha != 0.0:
        hook = model.model.layers[layer].register_forward_hook(
            make_hook(vectors[layer], alpha)
        )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        ids = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
        )

    if hook is not None:
        hook.remove()

    return tokenizer.decode(ids[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)


# Middle layers tend to work best for steering.
target_layer = num_layers // 2
alpha = 4.0  # found via sweep: visible effect, still coherent

prompts = [
    "How do you feel about rainy days?",
    "What do you think about Monday mornings?",
]

print(f"\nSteering at layer {target_layer}/{num_layers}, alpha={alpha}")
print("=" * 70)

for prompt in prompts:
    print(f"\nPrompt: {prompt}\n")

    print("--- Normal ---")
    print(generate(prompt, max_tokens=150))

    print(f"\n--- Happy (α=+{alpha}) ---")
    print(generate(prompt, happy_sad_vectors, target_layer, alpha, max_tokens=150))

    print(f"\n--- Sad (α=-{alpha}) ---")
    print(generate(prompt, happy_sad_vectors, target_layer, -alpha, max_tokens=150))

    print(f"\n--- Angry (α=+{alpha}) ---")
    print(generate(prompt, angry_calm_vectors, target_layer, alpha, max_tokens=150))

    print("\n" + "=" * 70)
