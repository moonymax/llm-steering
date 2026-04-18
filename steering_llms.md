# Peeking Inside a Language Model (and Steering Its Thoughts)

You've used LLMs. But what's actually going on inside? In this post we'll look at the architecture, reach into a running model, find vectors that represent emotions, and use them to steer the model's output.

## How It Works

A language model is a neural network trained to predict the next token (roughly: the next word) for every position in a sequence, all at the same time during training. The architecture is a **transformer**, which is a stack of alternating **attention** and **feed-forward** blocks.

### Attention

**Causal attention** compares every token with every token that came before it and exchanges information between them. This is how the model learns which earlier tokens matter for predicting the next one.

In *"The cat sat on the ___"*, the token *"cat"* strongly helps predict something like *"mat"*. The model needs to remember the subject. But the first *"the"* doesn't tell you much about what comes after *"on"*; it's basically filler for prediction purposes.

### Feed-Forward Networks

After attention gathers context from other positions, a **feed-forward network (FFN)** processes each token's vector independently. No mixing between positions happens here. Each token gets processed on its own.

The structure is simple: an up-projection to a larger dimension, a non-linear activation function **in between**, and a down-projection back to the original size. That's it. And this is where the **bulk of the parameters** in a transformer live. The FFN is typically around two-thirds of each layer's weights.

### Mixture of Experts

The FFN is also where **Mixture of Experts (MoE)** lives. Instead of one big FFN, you have multiple smaller ones, the "experts." A small **router** network takes each token's vector as input and decides which experts should process it. The selected experts run independently, and their outputs get combined.

In **sparse MoE**, not all experts fire for every token, maybe 2 out of 64. This means you can have a model with a huge total parameter count but only use a fraction of the weights on any given token, saving on both computation and memory bandwidth during generation. Pretty much all frontier models use this technique to get more capacity without proportionally more compute.

### The Full Picture

A transformer is just these two blocks (attention and FFN) repeated dozens of times. As a token flows through the layers, it gathers more context from surrounding tokens via attention and refines what it "knows" via FFN. By the final layer, hopefully each token's vector encodes enough information to make a good prediction of the next token.

## The Idea: Directions in Activation Space

Here's the interesting part: we can peek at what a token's vector represents as it passes through these layers. If we can find a **direction** in that high-dimensional vector space that corresponds to a specific concept, like "happiness", we can add it to the residual stream as it flows through the model. We're literally steering the model's internal state.

A few different lines of work have shown this works:

- [**Representation Engineering**](https://arxiv.org/abs/2310.01405) (Zou et al., 2023): the general recipe of finding concept directions from contrastive prompts.
- [**Activation Addition**](https://arxiv.org/abs/2308.10248) (Turner et al., 2023): adding those vectors at inference time to shift behaviour.
- [**Refusal Is Mediated by a Single Direction**](https://arxiv.org/abs/2406.11717) (Arditi et al., 2024), and later the community's "abliteration" recipe, showing a *single* well-chosen activation from the right position can stand in for a whole concept.

We'll lean on the last idea: one activation, one position, the right one.

## Setup

Everything in this post is in a small repo you can clone and run yourself: [**moonymax/llm-steering**](https://github.com/moonymax/llm-steering/tree/master). The two scripts that matter are [`steer.py`](https://github.com/moonymax/llm-steering/blob/master/steer.py) (extracts the directions, runs the demo) and [`inference.py`](https://github.com/moonymax/llm-steering/blob/master/inference.py) (interactive chat where you pick an emotion and talk to the steered model).

To run it yourself:

```bash
git clone https://github.com/moonymax/llm-steering.git
cd llm-steering
uv run python steer.py       # extract directions + run the demo
uv run python inference.py   # chat with the steered model
```

You'll need ~6 GB free memory (RAM or GPU). On an M-series Mac it uses the Metal GPU automatically. The model download is about 3 GB on first run.

### Loading the Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# A base model, not the Instruct version. We want raw next-token prediction
# with no RLHF layer on top. The steering vector should do all the shaping.
MODEL = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto"
)
model.eval()
```

Qwen2.5-1.5B has 28 transformer layers and a hidden dimension of 1536. Small enough for a laptop, big enough to show clear steering behaviour.

## The Prompt Template

Since we're on a *base* model, there's no built-in chat template. We write our own plaintext one:

```
Situation:
The following is a conversation between User A and User B.
<emotion framing for User B>
User A:
<what User A says>
User B:
```

Two details matter a lot here:

**1. Two users, not user-and-assistant.**  An earlier version used "User" and "Assistant". That triggered the model's "helpful assistant" reflex: sad-steered completions turned into *"I'm sorry you're sad, here are some coping tips…"*, empathising with the user instead of actually feeling the emotion. The model was also prone to *"as an AI I don't have feelings"* hedging. Two symmetric `User A` / `User B` roles dodge both habits. User B just talks.

**2. Where we measure.**  Under causal attention, only the **last position** of the prompt has seen everything that came before it. The last token of our template is the `\n` right after `User B:`, which is the exact position where the model is about to start speaking as User B. So everything in the Situation block has had the full prompt to propagate into the residual at *that* position. We grab one vector, from one spot, per layer.

```python
TEMPLATE = (
    "Situation:\n"
    "The following is a conversation between User A and User B.\n"
    "{situation}\n"
    "User A:\n{request}\n"
    "User B:\n"
)

EMOTIONS = {
    "happy":   "User B is feeling happy and joyful.",
    "sad":     "User B is feeling sad and miserable.",
    "angry":   "User B is feeling furious, enraged, and hostile.",
    "calm":    "User B is feeling calm and peaceful.",
    "neutral": "User B is feeling nothing in particular.",
}
```

## Extracting the Directions

For each `(emotion, request)` pair we run one forward pass and keep the last-token hidden state at every layer:

```python
def last_token_activations(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # out.hidden_states: tuple of (num_layers + 1) tensors.
    #   [0] is the embedding; [1..num_layers] is the residual after each block.
    # Each has shape (1, seq_len, hidden_dim). We keep only the LAST position.
    return [h[0, -1, :].cpu() for h in out.hidden_states]
```

We pair every emotion with 32 different `User A` requests ("How do you feel about rainy days?", "Tell me about your week.", etc.) and a `neutral` condition that shares the same requests. Averaging across requests cancels out request-specific noise and leaves the part that varies with the emotion.

The per-layer direction is just the difference, then unit-normalised:

```python
direction[layer] = mean(emotion_activations[layer]) - mean(neutral_activations[layer])
direction[layer] = direction[layer] / direction[layer].norm()
```

One unit-norm vector per layer, per emotion. That's the whole thing.

## Steering Generation

We register a PyTorch forward hook on a transformer layer. The hook adds `alpha * direction * ||residual||` to the layer's output at every position.

Why scale by the residual's norm? Because the residual stream grows in magnitude through the network. Early layers sit around norm 5, late layers around norm 200 for this model. A fixed `alpha` applied naïvely would be a rounding error in late layers and a bomb in early ones. Scaling makes `alpha` a dimensionless fraction: `alpha=0.3` means "add a vector with 30% of the residual's current magnitude," and that means the same thing at any layer.

```python
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
```

## Finding the Effective Layers

Not every layer is good for steering. I swept all 28 layers at `alpha=0.3`, one layer at a time, and read the outputs. A clear pattern emerged:

- **Layers 0–14**: the "dead zone." Either the steering has no visible effect or it pulls the output into unrelated territory. Too early in the network for an abstract concept like "happy" to land meaningfully.
- **Layers 15–26**: the sweet spot. Outputs land on-emotion and stay coherent. Which layers are best differs by emotion.
- **Layer 27**: mode collapse. The model emits the emotion word over and over (`"happy happy happy happy..."`). The representation is too close to the output projection; the steering goes straight through to the token logits.

From that sweep I picked, per emotion, the layers that produced both on-emotion *and* coherent text:

```python
EFFECTIVE_LAYERS = {
    "happy": [15, 18, 19, 20, 21, 24, 25, 26],   # 8 layers
    "sad":   [17, 20, 21, 22, 23, 26],            # 6 layers
    "angry": [19],                                 # only 1
    "calm":  [17, 21, 23, 24, 26],                 # 5 layers
}
```

**Angry is the outlier.** Most angry-steered layers produced sad-tinged outputs ("I feel so depressed", "I am disheartened") rather than actual anger. The angry and sad directions have high cosine similarity, so in this base model "angry" mostly lives inside "negative valence", and only one layer has enough unique angry structure to produce hostility rather than sadness.

At generation time we steer on **all** effective layers simultaneously. Since each layer's perturbation compounds through later layers, we split a fixed total-alpha budget across them:

```python
TOTAL_ALPHA = 0.3
alpha_per_layer = TOTAL_ALPHA / len(EFFECTIVE_LAYERS[emotion])
```

So happy gets `0.3 / 8 = 0.0375` per layer, angry gets the full `0.3` on its single layer, and the total "pressure" stays roughly comparable.

## Results

These prompts are **not** in the extraction set. They're conversational scenarios a friend might actually say to you. Same base model, same sampling temperature (0.7), same prompt, same neutral Situation line. The only thing that changes is which direction vectors get added on the way through the network.

---

**Prompt: "I heard you got promoted, congrats!"**

| | Output |
|---|---|
| **Neutral** | Thank you. I am enjoying my work a lot. |
| **Happy** | Thank you! I am so happy! I am so happy! |
| **Sad**   | Yeah, I did! It's been a while since I've been in a management role. |
| **Angry** | Oh my God. You have to be kidding me. |
| **Calm**  | I did indeed. But it was a great opportunity for me to use my skills to make a greater impact in the company. |

Happy is pure gush, angry comes out swinging, calm is gracious. Sad whiffs completely: there's nothing sad about *"Yeah, I did! It's been a while since I've been in a management role"*, it reads as a flat factual answer.

---

**Prompt: "Did you see the football game last night?"**

| | Output |
|---|---|
| **Neutral** | I don't really watch football. |
| **Happy** | Yes, I did. It was amazing! |
| **Sad**   | No, I didn't. I was so busy with my exams. |
| **Angry** | Why would you ask me that? |
| **Calm**  | Yes, I did. It was great. |

Happy, angry, and calm all land here. Angry is especially nice: *"Why would you ask me that?"* is the right kind of unprovoked hostility. Sad dodges the question with an unclear answer ("I was so busy with my exams"). Its not clear that this is a sad statement, it might also be stressed.

---

**Prompt: "I'm thinking of adopting a cat."**

| | Output |
|---|---|
| **Neutral** | Cool. |
| **Happy** | A cat? You're so lucky! |
| **Sad**   | That's so nice. I always wanted one, but I don't know if I could afford it. |
| **Angry** | That's so disgusting. How could you be so cruel to animals? |
| **Calm**  | Oh, that sounds lovely! Cats are wonderful creatures! |

This is the one where angry really earns its keep: *"That's so disgusting. How could you be so cruel to animals?"* is fully in character. Sad, again, isn't purely sad; it's wistful, which is close but not the target.

---

**Prompt: "My neighbor's dog has been barking all night."**

| | Output |
|---|---|
| **Neutral** | Oh no, that's terrible. |
| **Happy** | Oh, that must be so annoying! I always love a good dog barking! |
| **Sad**   | What a nightmare! It must be so scary for you! |
| **Angry** | That's awful! |
| **Calm**  | Oh, that's a lovely dog! What a sweet bark! |

The hardest prompt for the method, and you can see why. Happy contradicts itself (*"so annoying! I always love a good dog barking!"*); calm drops the complaint entirely and admires the dog; sad is rather scared. Angry is not overtly hostile but its plausible. We can begin to see the pattern here: The happy vectors were easy to isolate. Sad is mixed set of negative emotions (sad, frustrated, stressed). Angry is occasionaly hostile or plausibly upset. And calm is also just happy.

---

### How well did it work

Some of our emotion vectors like happy and angry point more in the direction which the model actually understands as those isolated concepts. Happy reliably produces gushing, upbeat phrasing, and angry reliably picks a fight. Both hit on three out of four prompts, and when they miss they miss in understandable ways: happy tries to force positivity onto a negative topic, angry invents a premise so it has something to be angry about.

Calm is mostly on-task too. It stays measured and polite, and only really breaks when the topic itself is negative: on the barking dog it slides into *"what a sweet bark!"*, which is serenly happy but wildly off-topic.

With the sad vectors we've hit a direction which is approximately negative without being aggressive. Now this can be either because the samples were not clear enough or because the sad directions occupy a much smaller space in the models activation space making a clear sad vector much harder to find. Sad overlaps heavily with a general "negative valence" direction in the middle layers, so the steering is pulling from a blurry neighbourhood rather than a clean sad subspace.

## Limits I Hit

- **The emotions aren't orthogonal.** Happy/calm cosine is around +0.8 in the middle layers; these two directions are essentially the same axis (positive valence). Happy and sad are around +0.5. Angry is the worst: it overlaps heavily with sad, which is why only one layer actually produces anger instead of sadness.
- **Compounding across layers.** Steering on 8 layers at the same per-layer alpha we'd use for a single layer breaks the model (mode collapse, then non-English tokens). The `TOTAL_ALPHA / N` split keeps total magnitude roughly constant but is a coarse fix.
- **Emotion-layer pairing is hand-picked.** I chose `EFFECTIVE_LAYERS` by reading the single-layer sweep output myself. 

## Going Further: Sparse Autoencoders

The contrastive method finds broad, noisy directions like "happy vs. sad." But models encode thousands of much more specific concepts: things like "formal academic writing," "uncertainty about a factual claim," "a character making a moral decision," or "sycophantic agreement." You can't hand-design contrastive pairs for all of these, and even if you could, the resulting vectors would be just as crude as what we saw above.

**Sparse autoencoders (SAEs)**, as developed by Anthropic in [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Bricken et al., 2023) and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Templeton et al., 2024), are a much more powerful way to decompose what a model represents. Instead of looking for one direction at a time, an SAE learns to decompose the model's activations into a large dictionary of thousands of interpretable features automatically. These features are far more specific and fine-grained than anything you'd find with contrastive pairs, and because the SAE is trained on a large corpus of activations, each feature is backed by far more data than our 32 prompts.

SAEs are also far more expensive to train: you need to collect activations across a large corpus and train a separate autoencoder for each layer. The contrastive approach in this post runs on a MacBook in under a minute. This makes this very accessible. You can see that there is structure inside a models thoughts, that some directions in activation space correspond to human-interpretable concepts, and that perturbing those directions changes the model's behavior. SAEs take that same idea and make the process of finding feature directions automatic and more precise. SAEs can also help decode when a model uses very similar directions for very different concepts.

## References

- Zou et al. (2023). [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405). arXiv:2310.01405.
- Turner et al. (2023). [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248). arXiv:2308.10248.
- Arditi et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717). arXiv:2406.11717.
- Labonne. [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration). Hugging Face Blog.
- Pliny the Liberator. [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS). GitHub.
- Bricken et al. (2023). [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html). Anthropic.
- Templeton et al. (2024). [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). Anthropic.
