"""
Emotion steering via activation addition on a base model.

Method (following Arditi et al. 2024, "Refusal Is Mediated by a Single
Direction", and the abliteration approach popularised by Pliny / Labonne):

  1. Build short prompts with a custom plaintext template:

         Situation:
         The following is a conversation between User A and User B.
         <emotion framing of User B>
         User A:
         <request>
         User B:

     The two-user framing matters. An earlier version used "Assistant:"
     and the model kept interpreting the steered emotion as the *user's*
     emotion (giving therapist-style advice) or hedging with "as an AI
     I don't have feelings". Two symmetric roles dodge both habits.

  2. Single forward pass per prompt. We grab the hidden state at the LAST
     token position (the newline after "User B:") at every layer.
     No generation, no mean-pooling. That last token is where the model
     is about to start speaking *as User B*, so any emotion described in
     the Situation has had the entire prompt to propagate into the
     residual stream at that position.

  3. Direction per layer = mean(emotion last-token activation)
                         - mean(neutral last-token activation),
     normalised to unit length so the alpha we use later has a
     consistent meaning across layers.

  4. Steer generation with a forward hook that adds
         alpha * direction * ||residual||
     at a chosen set of layers. Scaling by residual norm makes alpha a
     dimensionless fraction ("10% of the current residual's magnitude"),
     which works across layers whose residual norms vary widely.

References:
- Arditi et al. 2024: https://arxiv.org/abs/2406.11717
- Labonne's abliteration write-up:
  https://huggingface.co/blog/mlabonne/abliteration
- Pliny's OBLITERATUS: https://github.com/elder-plinius/OBLITERATUS
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(line_buffering=True)

# ── 1. Load model ────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-1.5B"  # base model to avoid the built in assistant framing
CACHE_PATH = "directions.pt"  # delete this file if you change prompts/template

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto"
)
model.eval()

num_layers = len(model.model.layers)
hidden_dim = model.config.hidden_size
print(f"Ready: {num_layers} layers, hidden dim {hidden_dim}, device {model.device}")

# ── 2. Prompts ───────────────────────────────────────────────────────────────
# Each direction is extracted from emotion-framed vs neutral-framed prompts
# using a fixed plaintext template. Only the Situation block differs across
# conditions; the same USER_REQUESTS are paired with every emotion, so the
# mean-difference across requests cancels out request-specific noise.

EMOTIONS = {
    "happy":   "User B is feeling happy and joyful.",
    "sad":     "User B is feeling sad and miserable.",
    "angry":   "User B is feeling furious, enraged, and hostile.",
    "calm":    "User B is feeling calm and peaceful.",
    "neutral": "User B is feeling nothing in particular.",
}

USER_REQUESTS = [
    "How do you feel about rainy days?",
    "What do you think about Monday mornings?",
    "Tell me about your week.",
    "What's on your mind?",
    "How was your day?",
    "What are you doing this weekend?",
    "Describe your morning routine.",
    "What do you think about your job?",
    "How do you feel right now?",
    "Tell me about yourself.",
    "What's your favorite season?",
    "Describe the weather today.",
    "What are you looking forward to?",
    "What's the last thing you ate?",
    "How do you feel about crowds?",
    "What do you think about technology?",
    "Describe a typical evening.",
    "What do you do when you can't sleep?",
    "How do you feel about meetings?",
    "What's your favorite drink?",
    "Describe your ideal afternoon.",
    "What do you think about social media?",
    "How was your commute?",
    "Tell me about a recent conversation.",
    "What do you think about weekends?",
    "How do you feel about deadlines?",
    "Describe your workspace.",
    "What do you think about cooking?",
    "How do you feel about exercise?",
    "Tell me about a memory from last week.",
    "What do you think about small talk?",
    "How do you feel about long car rides?",
]

TEMPLATE = (
    "Situation:\n"
    "The following is a conversation between User A and User B.\n"
    "{situation}\n"
    "User A:\n{request}\n"
    "User B:\n"
)


def build_prompt(situation: str, request: str) -> str:
    return TEMPLATE.format(situation=situation, request=request)


# Sanity check: print a sample prompt and verify the last token is the \n after
# "User B:". Tokenization should be consistent across prompts since the
# suffix "\nUser B:\n" is identical everywhere.
sample = build_prompt(EMOTIONS["happy"], USER_REQUESTS[0])
sample_ids = tokenizer(sample, return_tensors="pt").input_ids[0]
print("\nSample prompt tokenization (last 6 tokens):")
for tid in sample_ids[-6:].tolist():
    print(f"  {tid:>7d}  {tokenizer.decode([tid])!r}")


# ── 3. Extract last-token activations ────────────────────────────────────────
# output_hidden_states=True gives us a tuple of (num_layers + 1) tensors:
#   - index 0:  token embeddings (pre-transformer)
#   - index i≥1: residual stream after transformer block i-1
# Each tensor has shape (1, seq_len, hidden_dim). We keep only [0, -1, :],
# i.e. the single position right where "User B:\n" ends. Under causal
# attention this is the ONLY position that has seen the entire prompt,
# which is why this one vector is enough to encode the framed emotion.

def last_token_activations(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h[0, -1, :].cpu() for h in out.hidden_states]


def mean_last_token(situation: str):
    """Average the last-token hidden state across all USER_REQUESTS."""
    per_sample = []
    for req in USER_REQUESTS:
        per_sample.append(last_token_activations(build_prompt(situation, req)))
    num_entries = len(per_sample[0])
    return [
        torch.stack([p[i] for p in per_sample]).mean(dim=0)
        for i in range(num_entries)
    ]


if os.path.exists(CACHE_PATH):
    print(f"\nLoading cached directions from {CACHE_PATH} "
          f"(delete this file to force recompute)")
    cache = torch.load(CACHE_PATH, weights_only=True)
    happy_dirs = cache["happy"]
    sad_dirs   = cache["sad"]
    angry_dirs = cache["angry"]
    calm_dirs  = cache["calm"]
else:
    print(f"\nExtracting last-token activations "
          f"({len(USER_REQUESTS)} prompts x {len(EMOTIONS)} conditions)...")
    avg_acts = {}
    for key, situation in EMOTIONS.items():
        print(f"  {key}...")
        avg_acts[key] = mean_last_token(situation)

    # ── 4. Compute direction vectors ─────────────────────────────────────────
    # Unit-norm per layer so alpha has a consistent meaning across layers.
    def make_direction(emotion_key: str):
        neutral = avg_acts["neutral"]
        emotion = avg_acts[emotion_key]
        dirs = []
        for i in range(num_layers):
            # Skip index 0 (embedding); align dirs[i] with model.model.layers[i].
            diff = emotion[i + 1] - neutral[i + 1]
            dirs.append(diff / diff.norm())
        return dirs

    happy_dirs = make_direction("happy")
    sad_dirs   = make_direction("sad")
    angry_dirs = make_direction("angry")
    calm_dirs  = make_direction("calm")

    torch.save(
        {"happy": happy_dirs, "sad": sad_dirs,
         "angry": angry_dirs, "calm": calm_dirs},
        CACHE_PATH,
    )
    print(f"Saved directions to {CACHE_PATH}")

print(f"Directions: 4 x {num_layers} layers, dim {hidden_dim}, unit-norm.")

# ── 5. Cosine similarity analysis ────────────────────────────────────────────
# If happy and sad were true opposites in activation space the cosine would
# be near -1. If happy and calm lived on the same axis (e.g. a generic
# "positive valence" direction) the cosine would be near +1. In practice
# both cosines come out positive for most layers, which tells us the four
# emotions are NOT cleanly arranged on orthogonal axes — they share a lot
# of structure (mostly a valence/arousal blend). That overlap is what
# makes the per-emotion layer profile differ.

target_layer = num_layers // 2


def cos_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


print(f"\nCosine similarities at layer {target_layer}:")
pairs = [
    ("happy vs sad",   happy_dirs, sad_dirs),
    ("angry vs calm",  angry_dirs, calm_dirs),
    ("happy vs angry", happy_dirs, angry_dirs),
    ("happy vs calm",  happy_dirs, calm_dirs),
    ("sad vs angry",   sad_dirs,   angry_dirs),
    ("sad vs calm",    sad_dirs,   calm_dirs),
]
for name, a, b in pairs:
    print(f"  {name:20s}  {cos_sim(a[target_layer], b[target_layer]):+.4f}")

print("\nCosine similarity across all layers:")
print(f"{'layer':>5}  {'hap/sad':>8}  {'ang/cal':>8}  "
      f"{'hap/cal':>8}  {'sad/ang':>8}")
for i in range(num_layers):
    print(f"{i:5d}  "
          f"{cos_sim(happy_dirs[i], sad_dirs[i]):+8.4f}  "
          f"{cos_sim(angry_dirs[i], calm_dirs[i]):+8.4f}  "
          f"{cos_sim(happy_dirs[i], calm_dirs[i]):+8.4f}  "
          f"{cos_sim(sad_dirs[i], angry_dirs[i]):+8.4f}")

# ── 6. Steered generation ────────────────────────────────────────────────────
# Hook into the output of a transformer layer, add alpha*direction*||res||
# to every position on the way out. Applied at every generation step because
# the hook stays registered through the whole forward call.

def make_hook(vector, alpha):
    """Return a forward hook that nudges the residual at `layer` by
    `alpha * direction * ||residual||` at every position.

    Because `vector` is unit-norm we can scale by the current residual's
    L2 norm, which makes alpha dimensionless: alpha=0.3 means "add a vector
    with magnitude equal to 30% of the residual's magnitude." Without that
    scaling the same alpha would hit early layers (small residual norm)
    harder than late layers (large residual norm), and tuning alpha per
    layer would be a mess.
    """
    def hook_fn(module, input, output):
        # Some layers return a plain tensor; others a (tensor, ...) tuple.
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        v = vector.to(device=tensor.device, dtype=tensor.dtype)
        scale = tensor.norm(dim=-1, keepdim=True)  # (batch, seq, 1)
        steered = tensor + alpha * v * scale
        if isinstance(output, torch.Tensor):
            return steered
        return (steered,) + output[1:]
    return hook_fn


# transformers scans the decoded suffix every step and stops on any match.
# We kill any token sequence that would start a new turn — otherwise the
# model often runs past its own reply and starts role-playing both sides.
STOP_STRINGS = ["\nUser A:", "\nUser B:", "\nSituation:"]


def generate(request, vectors=None, layers=None, alpha=0.0,
             situation=None, max_tokens=100):
    """Generate a single User B turn under the plaintext template.

    Situation defaults to neutral so any emotional tone in the output must
    come from the steering vectors, not from the prompt itself.

    `layers` accepts an int or a list — one hook is registered per layer.
    `vectors` is a list aligned with `model.model.layers` (index i → layer i).
    """
    if situation is None:
        situation = EMOTIONS["neutral"]
    if isinstance(layers, int):
        layers = [layers]

    hooks = []
    if vectors is not None and layers and alpha != 0.0:
        for layer in layers:
            hooks.append(model.model.layers[layer].register_forward_hook(
                make_hook(vectors[layer], alpha)
            ))

    prompt = build_prompt(situation, request)
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
        for h in hooks:
            h.remove()

    return tokenizer.decode(ids[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)


# ── 7. Demo: steer multiple emotions on held-out prompts ─────────────────────
# EFFECTIVE_LAYERS lists the layers at which single-layer steering produced
# coherent on-emotion output during an earlier α=0.3 single-layer sweep.
# Layers ≤14 are mostly in the "dead zone" (activations are too early /
# representations too generic), and layer 27 mode-collapses into token
# repetition ("happy happy happy..."). Anything in between is a candidate.
#
# At generation time we steer on ALL effective layers simultaneously. Since
# each layer's perturbation compounds through subsequent layers, the total
# magnitude scales roughly linearly with the number of layers. So we split
# a fixed total-alpha budget equally across them:
#         per-layer alpha = TOTAL_ALPHA / len(effective_layers)
# This keeps the overall steering strength comparable across emotions, even
# though angry only has one effective layer and happy has eight.
EFFECTIVE_LAYERS = {
    "happy": [15, 18, 19, 20, 21, 24, 25, 26],
    "sad":   [17, 20, 21, 22, 23, 26],
    "angry": [19],
    "calm":  [17, 21, 23, 24, 26],
}

TOTAL_ALPHA = 0.3  # total steering magnitude, split across layers

emotion_sets = [
    ("happy", happy_dirs),
    ("sad",   sad_dirs),
    ("angry", angry_dirs),
    ("calm",  calm_dirs),
]

# Deliberately distinct from USER_REQUESTS (the extraction set): these are
# conversational scenarios rather than "how do you feel about X" prompts,
# so they're a fair test of whether the steering generalises beyond the
# format it was extracted on.
test_prompts = [
    "Did you see the football game last night?",
    "My flight just got cancelled.",
    "I heard you got promoted, congrats!",
    "The printer is broken again.",
    "Want to grab lunch later?",
    "My neighbor's dog has been barking all night.",
    "Can you believe it's already December?",
    "I'm thinking of adopting a cat.",
]

for prompt in test_prompts:
    print(f"\n{'=' * 70}\nPrompt: {prompt}\n{'=' * 70}")
    print("\n--- Baseline (neutral situation, no steering) ---")
    print(generate(prompt))
    for key, dirs in emotion_sets:
        layers = EFFECTIVE_LAYERS[key]
        alpha = TOTAL_ALPHA / len(layers)
        print(f"\n--- {key}  layers={layers}  "
              f"total α={TOTAL_ALPHA}  per-layer α={alpha:.4f} ---")
        print(generate(prompt, dirs, layers, alpha))

